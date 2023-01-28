# Bird-Species-Audio-Identification-Tool


### what is the problem PLACEHOLDER SUBTITLE

Whilst working with an ecological field station, the issue of having years of (good!) audio data but no capacity to analyse it. It would take hundreds of volunteers to listen to these recordings and identify which species were present by any meaningful deadline. Furthermore, any volunteer would have to invest time training to identify calls (of which there may be a dizzying number), after which there is no guarantee of accuracy due to a range of factors - such as background noise from weather dramatically affecting what can be accurately heard.

Whilst it may be easier for volunteers to identify animals within images, audio recordings provide the advantage of capturing the activity of animals in a 360 degree area around the device, and can capture this activity even if the animal is not in direct line of sight (for example, due to trees or bushes, or during night hours) 

Without knowing what species are on the island it would be difficult to apply for funding to continue conservation efforts, and almost impossible to build a case for protecting the land against rapidly enroaching aquaculture and tourism expansion.

### What is the solution

To solve the issue of volunteer capacity, both in terms of 'labour pool' and of identification skill, a deep learning neural net audio classification tool was developed within Python (using Tensorflow). Audio was transformed into a waveform (an image representation of sound - see below for more information) format with which object detection was carried out on the long survey audio files. 

The audio classification tools were trained on volunteer created training libraries (produced after leading a short [Audacity](https://www.audacityteam.org/) easily delivered and enthusiastically understood workshop. Audio clips were sourced from Xenocanto, eBird, and even YouTube, meaning the volunteer doesn't need to learn or apply identification skills, just a short clipping process which many found enjoyable.

As the tool is intended to be utilised long after my departure, the Python script is intended to be run with as little interaction as possible (hence the folder and file checks to avoid errors), so that any volunteer can open Spyder and press run

Please note that the tool doesn't account for imperfect detection (for example, a bird at 10m may be heard 100% of the time, but a bird at 40m may only be heard 8% of the time), and that the data presented on this repository is both _abbreviated and anonymised_ (as some species are incredibly rare, and poaching activity is well known in the area!)

## placeholder title

Define species richness & biodiversity, and describe trend/caveats

define how we're calculating population size:

_grab those papers for the actual definition_

![Copy of Copy of Bottom 3 (3)](https://user-images.githubusercontent.com/122735369/215264531-1eb4be82-d189-430c-9fc9-6dd15f8962fa.png)

<p align="center"><sup>placeholder caption - abbreviated table - again explain anonymisation</sup></p>

Describe relationships - something like - those in smaller groups tend to heard to make fewer calls both per individual and in total

![Bottom 3](https://user-images.githubusercontent.com/122735369/215263543-9525ba62-15c4-4f82-a8e2-3bbc7b28d916.png)

<p align="center"><sup>placeholder caption</sup></p>

## placeholder title

explain 'under the hood' for a non technical audience. explain the graph is to identify which species need more files for DL model training
recall - What proportion of actual positives was identified correctly? (more important - we want to know if the species was there)
precision - What proportion of positive identifications was actually correct? (not AS bad if we say that a species was there when it wasn't)
Accuracy is generally linked inversely to loss, so we want as little as possible 

![Copy of Copy of Bottom 3 (2)](https://user-images.githubusercontent.com/122735369/215263692-78bd5a51-9120-4ff6-b6d3-6f26c6bbf132.png)
<p align="center"><sup>placeholder caption</sup></p>

## next steps



