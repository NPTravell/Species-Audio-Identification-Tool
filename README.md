# Bird-Species-Audio-Identification-Tool


### What is the problem with wildlife audio data?

Whilst working with an ecological field station, we had years of (good!) audio data but no capacity to analyse it. It would take hundreds of volunteers manually identifying which bird species were recorded within to finish by any meaningful deadline. Even after investing time in identification training, various factors prevent a guarantee of accuracy - such as background weather noise affecting what, if anything, can be heard.

Image identification _may_ be easier for volunteers, however audio recorders can capture bird activity in a 360 degree arc, including animals not in direct line of sight (such as when sitting in covered trees, or calling out during the night) 

Without knowing what species are on the island it would be difficult to continue funding conservation efforts, or to build a case for protecting the land against rapidly enroaching aquaculture and tourism expansion.

### What was the solution?

To solve the issue of volunteer 'labour pool' and identification skill, a 'audio classification' tool was developed within Python (using Tensorflow). Audio was transformed into a waveform image with which object detection (via a 'deep learning neural network') was carried out on the long survey audio files. 

Volunteers created training 'sound reference libraries' (produced after leading a short, easily delivered, and enthusiastically understood [Audacity](https://www.audacityteam.org/) workshop). Audio clips were sourced from Xenocanto, eBird, and even YouTube, meaning the volunteer didn't need to learn any identification skills, just the short clipping process. Background noise files (such as human activity, rainfall, terrestrial mammal calls) are also involved in model training.

As the tool is intended to be utilised long after my departure, the Python script is intended to be run with as little interaction as possible. The script will produce a seperate deep learning neural network model for each species that is beleived to be on the island, train on the sound reference libraries, and analyse long-term survey audio. The final result is an Excel file detailing what species was heard in each recording, how many times the species was heard, and what the estimated population is.

Note: the tool doesn't account for imperfect detection (for example, a bird at 10m may be heard 100% of the time, but a bird at 40m may only be heard 8% of the time - which would result in undercounting). Note also that the data presented on this repository is anonymised as some species are incredibly rare, and poaching activity is well known in the area

## Deep learning neural network audio classification

The tool splits long survey audio up into shorter clips and uses object detection to attempt to answer the question "does this survey waveform _look_ like the species waveform?"

A graph is supplied to show the field station which species have models that are underperforming (and thus may require more reference sound files for training). A full understanding of precision (which is the proportion of positive identifications for a species that were correct), and recall (proportion of real positives of a species that were identified correctly) are not required - simply knowing where any big performance dips, or loss spikes, occur - or even a gap indicating a model was not trained at all - will suffice.

![Copy of Copy of Bottom 3 (2)](https://user-images.githubusercontent.com/122735369/215263692-78bd5a51-9120-4ff6-b6d3-6f26c6bbf132.png)
<p align="center"><sup>Left: Example waveform comparison between a bird call and background noise. Right: Chart showing how the trained models compare</sup></p>

## Next Steps

The tool has already been implemented at the field station, and processes established to both allow untrained volunteers to generate sound reference libraries for model training, as well as to identify species in survey recordings brought back from the field. The next steps are recommended:

1. Comparisons between the models performance and more pen and paper identification methods should be carried out to confirm the tool is working as intended and at least as good as manual volunteer assessment.

2. As the computer hardware on site was limited, relatively small models (with fewer and smaller neuronal layers) were utilised. Perhaps efforts to at least temporarily acquire more powerful hardware and to experiment with more complex and larger architecture could be made. 

## What did we discover about bird species on the island?

Species richness is the count of the unique species, and is an important measure of biodiversity. The table below is abbreviated to only include 3 (anonymised) locations across 3 months in 2017, in which we can see fairly static species richness during this time. This could be due to the island context (with migration to and from the location being limited to those with the capacity to cross the sea) or due to the particular season(s) selected - with colder months possibly being periods of relatively lower activity.

![Copy of Copy of Bottom 3 (3)](https://user-images.githubusercontent.com/122735369/215264531-1eb4be82-d189-430c-9fc9-6dd15f8962fa.png)

<p align="center"><sup>Left: Table showing the unique count of species heard and identified over time. Right: Example view from where one audio detector was placed</sup></p>

Along with knowing which species are heard, it can be useful to know how many individuals are present, however it can be difficult to discern how many individuals are contributing to this call tally.

Scientific literature suggests that a method of estimating abundance could be to divide the total number of cues detected per unit time by the species average cue rate. The following abbreviated tables show the results from this method, and a relationship between those of smaller group size and lower call rates as well as fewer total calls was established. Individuals of larger groups (species 12, 18, and 17) tended to be recorded more in total, but were not identified to be the species with the most calls per individuals; suggesting that more species with more central maximum group sizes had individuals that called more often

![Bottom 3](https://user-images.githubusercontent.com/122735369/215263543-9525ba62-15c4-4f82-a8e2-3bbc7b28d916.png)

<p align="center"><sup>Charts showing trends in various aspects of call behaviour</sup></p>

<p align="center"><sup> This code was produced to help a wildlife charity - visitors are encouraged to ask questions and/or expand upon the script with additional features (particularly regarding model architecture, enhanced visualisations, and improving work flow/process checks). If any research organisations want help implementing, or a free demo of, the tool then feel free to reach out</sup></p>
