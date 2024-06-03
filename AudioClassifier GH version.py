"""
Deep audio classifier built with Tensorflow
Uses sound libraries to train detection models and identify species within survey audio recordings

"""

import os
import pandas as pd
import csv
from matplotlib import pyplot as plt
from itertools import groupby
import tensorflow as tf #installed version 2.8.0, needed protobuf==3.20.*
import tensorflow_io as tfio #installed version 0.24.0
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten
from datetime import datetime
from mutagen.mp3 import MP3
import math
import numpy as np
import random
import calendar

#set wd
os.chdir('') #removed for GitHub

#grab list of species, check how many species are in there
Species_list=pd.read_csv('Species_list.csv')
len(Species_list)

#check that reference library exists for each species  and check how many files are in each
for i in range(len(Species_list)):
    species=Species_list.loc[i, "Species"]
    print(str(species) + ': '+ str(os.path.exists('./Sound_reference_library/'+species)))
    if os.path.exists('./Sound_reference_library/'+species) == True:
        if len(os.listdir('./Sound_reference_library/'+species)) > 0:
            print(str(len(os.listdir('./Sound_reference_library/'+species)))+' files')
del(i,species)

#define dataloading function (taken from tensorflow tf.io documentation)  
def load_wav_16k_mono(filename):
    # Load encoded wav file. turns clip into decoded audio
    file_contents = tf.io.read_file(filename)
    # Decode wav (tensors by channels) 
    wav, sample_rate = tf.audio.decode_wav(file_contents, desired_channels=1)
    # Removes trailing axis
    wav = tf.squeeze(wav, axis=-1)
    sample_rate = tf.cast(sample_rate, dtype=tf.int64)
    # Goes from 44100Hz to 16000hz - amplitude of the audio signal
    wav = tfio.audio.resample(wav, rate_in=sample_rate, rate_out=16000)
    return wav

#define background noises common to the area that MAY be misidentified as a species
NEG = os.path.join('Sound_reference_library','background_noise')
neg = tf.data.Dataset.list_files(NEG+'/*.wav')

#plot example wave comparison (for Github - volunteers wouldn't need this)
Species_1_file = os.path.join('Sound_reference_library','Species_1')
background_noise_file = os.path.join('Sound_reference_library','background_noise')
nwave=load_wav_16k_mono('./'+background_noise_file+'/'+random.choice(os.listdir(background_noise_file)))
wave=load_wav_16k_mono('./'+Species_1_file+'/'+random.choice(os.listdir(Species_1_file)))

#plot the wave we returned above
plt.suptitle('What Do Background and Species Noise\nWaveforms Look Like?')
plt.plot(wave,label="Species 1 noise",color="blue")
plt.plot(nwave,label="Background noise",color="yellow")
plt.legend()
plt.savefig('waveform_example.png', bbox_inches='tight')
plt.show()
del(nwave,wave,Species_1_file,background_noise_file)

# =============================================================================
# #Model training
# =============================================================================

#define loading function
def preprocess(file_path, label): 
    wav = load_wav_16k_mono(file_path)
    wav = wav[:48000]
    #pads clips that aren't full
    zero_padding = tf.zeros([48000] - tf.shape(wav), dtype=tf.float32)
    #concats padded zeros with actual clip
    wav = tf.concat([zero_padding, wav],0)
    spectrogram = tf.signal.stft(wav, frame_length=320, frame_step=32)
    spectrogram = tf.abs(spectrogram)
    spectrogram = tf.expand_dims(spectrogram, axis=2)
    return spectrogram, label

#define a dataframe to add species model performance to
colnames=['Species','files','loss','recall','precision']
species_models=pd.DataFrame(columns=colnames)
del(colnames)

#loop to create models for each species sound reference library
#species='Species_4' #debugging
for i in range(len(Species_list)):
    species=Species_list.loc[i, "Species"]
    if os.path.exists('./Sound_reference_library/'+species) == True: #check reference folder exists for that species
        if len(os.listdir('./Sound_reference_library/'+species)) > 0: #check the folder isn't empty!
            if os.path.exists('./Models/'+species) == False: #check that a model hasn't already been trained for the species
                os.mkdir('./Models/'+species) #create a model directory
                
                #define location of species recordings and zip it with background noise
                POS = os.path.join('Sound_reference_library', species)
                pos = tf.data.Dataset.list_files(POS+'/*.wav')
                positives = tf.data.Dataset.zip((pos, tf.data.Dataset.from_tensor_slices(tf.ones(len(pos)))))
                negatives = tf.data.Dataset.zip((neg, tf.data.Dataset.from_tensor_slices(tf.zeros(len(neg)))))
                data = positives.concatenate(negatives)
                
                #partition training and test data
                data = data.map(preprocess)
                data = data.cache()
                data = data.shuffle(buffer_size=1000)
                data = data.batch(16)
                data = data.prefetch(8)
                train = data.take(36)
                test = data.skip(36).take(15)
                
                #build deep learning model #NOTE: this was the largest I could get on the supplied hardware
                model = Sequential()
                model.add(Conv2D(16, (3,3), activation='relu', input_shape=(1491, 257,1))) 
                model.add(Conv2D(16, (3,3), activation='relu')) 
                model.add(Flatten())
                model.add(Dense(128, activation='relu')) #128
                model.add(Dense(1, activation='sigmoid'))
                model.summary()
                model.compile('Adam', loss='BinaryCrossentropy', metrics=[tf.keras.metrics.Recall(),tf.keras.metrics.Precision()])
                            
                #fit the above specified model
                hist = model.fit(train, epochs=4, validation_data=test)
                
                #save the model
                model.save('./Models/'+str(species))
                
                #visualise performance
                plt.title('Loss')
                plt.plot(hist.history['loss'], 'r')
                plt.plot(hist.history['val_loss'], 'b')
                plt.savefig('./Models/'+str(species)+'/'+str(species)+'_loss.png', bbox_inches='tight')
                plt.show()
    
                plt.title('Precision')
                plt.plot(hist.history['precision'], 'r')
                plt.plot(hist.history['val_precision'], 'b')
                plt.savefig('./Models/'+str(species)+'/'+str(species)+'_precision.png', bbox_inches='tight')
                plt.show()
    
                plt.title('Recall')
                plt.plot(hist.history['recall'], 'r')
                plt.plot(hist.history['val_recall'], 'b')
                plt.savefig('./Models/'+str(species)+'/'+str(species)+'_recall.png', bbox_inches='tight')
                plt.show()
    
                #extract performance and save
                loss=hist.history['loss'][-1]
                recall=hist.history['recall'][-1]
                precision=hist.history['precision'][-1]
                files=len(os.listdir('./Sound_reference_library/'+species))
                species_models.loc[len(species_models)] = [species,files,loss,precision,recall]
del(POS,pos,NEG,neg,loss,hist,files,data,negatives,positives,precision,recall,train,test,species)

#save csv of model performances for review
species_models.to_csv('./species_models_performance.csv',index=False)
del(species_models)

#create visualisation to review model performances
model_performance=pd.read_csv('species_models_performance.csv')
model_performance.Species=model_performance.Species.str.split('_').str[1].astype(int)
model_performance=model_performance.sort_values('Species',ascending=True)
model_performance.recall=model_performance.recall*100
model_performance.precision=model_performance.precision*100

x=model_performance.Species
y=model_performance.recall
y2=model_performance.precision
y3=model_performance.loss

width = 0.5
fig = plt.figure()
gs = fig.add_gridspec(2, hspace=0,height_ratios=[2, 0.5])
axs = gs.subplots(sharex=True, sharey=False)
fig.suptitle('How Do Models Perform for\nEach Species?')
txt="*recall and precision displayed as a %"
fig.text(0.05,1,txt, fontsize=7)
axs[0].bar(x- 1* width/2, y,width,label="Recall",color="blue")
axs[0].bar(x+ 1*width/2, y2,width,label="Precision",color="cyan")
axs[1].bar(x- 1*width/2, y3,width,label="Loss",color="grey")
axs[1].invert_yaxis()
fig.legend(fontsize=6)
plt.xticks(x,fontsize=6)
plt.xlabel('Anonymised Species Model')
plt.show()
fig.savefig('./Model_performance.png', bbox_inches='tight')
del(model_performance,x,y,y2,y3,width,fig,gs,axs,txt)

# =============================================================================
# #Audio classification
# =============================================================================
                   
#define function to load a mp3 file and convert it #NOTE: would be faster if the recordings were wav in future
def load_mp3_16k_mono(filename):
    res = tfio.audio.AudioIOTensor(filename)
    tensor = res.to_tensor() 
    tensor = tf.math.reduce_sum(tensor, axis=1) / 2 
    sample_rate = res.rate
    sample_rate = tf.cast(sample_rate, dtype=tf.int64)
    wav = tfio.audio.resample(tensor, rate_in=sample_rate, rate_out=16000)
    return wav

#Function to convert clips to spectrogram for object detection
def preprocess_mp3(sample, index):
    sample = sample[0]
    zero_padding = tf.zeros([48000] - tf.shape(sample), dtype=tf.float32)
    wav = tf.concat([zero_padding, sample],0)
    spectrogram = tf.signal.stft(wav, frame_length=320, frame_step=32)
    spectrogram = tf.abs(spectrogram)
    spectrogram = tf.expand_dims(spectrogram, axis=2)
    return spectrogram

#loop over survey audio recordings and identify calls in each
#WARNING THIS TAKES A VERY LONG TIME - it needs to load each model, split up clips, detect the species in each clip, for every species in the list
results = {}
#species='Species_4' #debugging
for i in range(len(Species_list)):
    species=Species_list.loc[i, "Species"]
    print('Starting: '+str(species) +' at: ' + datetime.now().strftime("%H:%M:%S")) #for peace of mind
    if os.path.exists('./Models/'+str(species)) == True: #check there's a model for that species
        model = tf.keras.models.load_model('./Models/'+str(species)) #load that model 
        for file in os.listdir(os.path.join('Deployment_Recordings')):
            FILEPATH = os.path.join('Deployment_Recordings', file)
            wav = load_mp3_16k_mono(FILEPATH)
            
            #chop recordings into clips, turn the clips into images and run object detection
            audio_slices = tf.keras.utils.timeseries_dataset_from_array(wav, wav, sequence_length=48000, sequence_stride=48000, batch_size=1)
            audio_slices = audio_slices.map(preprocess_mp3)
            audio_slices = audio_slices.batch(2)
            yhat = model.predict(audio_slices)
            results[file] = yhat

            #Convert Predictions into Classes
            class_preds = {}
            for file, logits in results.items():
                class_preds[file] = [1 if prediction > 0.99 else 0 for prediction in logits]

            #group calls that are close together into one
            postprocessed = {}
            for file, scores in class_preds.items():
                postprocessed[file] = tf.math.reduce_sum([key for key, group in groupby(scores)]).numpy()
            
            #create results folder if it doesn't exist
            if os.path.exists('./Deployment_recordings_results') == False:
                os.mkdir('./Deployment_recordings_results')
            
            #write results csv into results folder
            with open('./Deployment_recordings_results/'+str(species)+'_results.csv', 'w', newline='') as f:
                writer = csv.writer(f, delimiter=',')
                writer.writerow(['recording', 'occurences'])
                for key, value in postprocessed.items():
                    writer.writerow([key, value])
del(i,wav,writer,yhat,postprocessed,species,Species_list,scores,results,audio_slices,class_preds,f,file,FILEPATH,key,logits,value)

#Population estimation using average cue rate and calls per unit time.
clip_lengths=pd.DataFrame(columns=['recording','length(s)']) #get lengths of each audio clip
for file in os.listdir('./Deployment_Recordings'):
    audio = MP3('Deployment_Recordings/'+file)
    audio=round(audio.info.length) 
    clip_lengths.loc[len(clip_lengths)] = [file,audio]
del(audio)

#loop through each results file, and calculate calls per unit time
#file='Species_17_results.csv' #debugging
#file='Species_25_results.csv' #debugging
all_results=pd.DataFrame()
for file in os.listdir('./Deployment_recordings_results'):  
    #open each results file
    results=pd.read_csv('Deployment_recordings_results/'+file)
    #merge with the above loop to get clip lengths
    results=pd.merge(results,clip_lengths, on="recording")
    #filter to clips where the species was heard
    results=results[results['occurences']>0]
    #divide the calls per second by the median species call rate (median calls per second), and round upwards (can't have a fraction of a bird, for example)
    results['estimated_population']=np.ceil((results['occurences']/results['length(s)'])/(results['occurences']/results['length(s)']).median())
    results=results.drop('length(s)',axis=1)
    #grab species name from file name
    species=file.split('.')[0]
    species=species.replace('results', '')    
    species=species.rstrip('_')
    results['species']=species
    all_results=all_results.append(results)
del(species,file,results)   

#pivot so that we have an easy to view table
all_results_pivoted = pd.pivot_table(all_results, values='estimated_population', index='recording', columns=['species'], aggfunc=np.sum)
del(clip_lengths)

#visualise table
fig, ax = plt.subplots()
fig.patch.set_visible(False)
ax.axis('off')
ax.axis('tight')
ax.table(cellText=all_results_pivoted.values, rowLabels=all_results_pivoted.index, colLabels=all_results_pivoted.columns, loc='center')
plt.rcParams['figure.dpi'] = 300
fig.savefig('./Species_population_results.png', bbox_inches='tight')
plt.show()
del(ax,fig)

#save csv files
all_results_pivoted.to_csv('Species_population_results_pivot.csv',index=False)
all_results.to_csv('Species_population_results.csv',index=False)
