# Species-Audio-Identification-Tool


Whilst working with an ecological field station, we had years of (good!) audio data but no capacity to analyse it to understand what species were there.

### How do we solve the issue of volunteer 'labour pool' and identification skill?

A deep learning object detection model was trained on audio libraries (sourced from sites like eBird or YouTube, and processed in [Audacity](https://www.audacityteam.org/)). The result was a solution that could answer the question "does this survey waveform _look_ like the species waveform?", and therefore tally up species calls and estimate population sizes

<p align="center"><sup>Note: the tool doesn't account for imperfect detection (for example, a species at 40m may only be heard 8% of the time - which would result in undercounting). Note also that the data presented on this repository is anonymised as some species are incredibly rare, and poaching activity is well known in the area</sup></p>

![Copy of Copy of Bottom 3 (2)](https://user-images.githubusercontent.com/122735369/215263692-78bd5a51-9120-4ff6-b6d3-6f26c6bbf132.png)
<p align="center"><sup>Left: Example waveform comparison between a bird call and background noise. Right: Chart showing how the trained models compare - highlighting models that need extra training files</sup></p>

## What are some of the insights we can discover?

Species richness, an important measure of biodiversity, is the count of the unique species. In the abbreviated table below we can see no change over time (possibly due to migration being limited on the island). **This metric can be used to evaluate conservation efforts, as well as providing evidence for policy lobbying.**

![Copy of Copy of Bottom 3 (3)](https://user-images.githubusercontent.com/122735369/215264531-1eb4be82-d189-430c-9fc9-6dd15f8962fa.png)

<p align="center"><sup>Right: Example view from where one audio detector was placed, and example anonymised species richness data</sup></p>

Individual species behaviour can be inferred from the results, however this requires domain knowledge to avoid incorrect conclusions. **Insights discovered here can be used for generating future research ideas or planning of future survey locations.**

![Bottom 3]![Screenshot 2024-06-03 124201](https://github.com/NPTravell/Species-Audio-Identification-Tool/assets/122735369/3c15113c-f47c-4f8b-adf5-f87a0d93c563)

<p align="center"><sup>Charts showing trends in various aspects of call behaviour. For example, species with the largest maximum group sizes did were seen to not have the highest call rates (possibly due to the risk of being be drowned out) </sup></p>

### Next Steps

The tool has been field tested with promising results, and the following steps are recommended to improve implementation:

1. Performance (accuracy and time spent) between the tool and manual identification methods should be compared

2. As the computer hardware on site was limited, models with small neuronal layers were utilised. Cloud software could be trialled for improved accuracy and performance

3. A simplified user interface can be created

<p align="center"><sup> This code was produced to help a wildlife charity - forks are encouraged that expand upon the script with additional features (particularly if they regard more efficient model architecture, enhanced visualisations, and/or improving work flow/process checks). If any organisations want help implementing this tool, feel free to reach out</sup></p>
