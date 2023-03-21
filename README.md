# Species-Audio-Identification-Tool


Whilst working with an ecological field station, we had years of (good!) audio data but no capacity to analyse it to understand what species were there.

### How do we solve the issue of volunteer 'labour pool' and identification skill?

Through using [Audacity](https://www.audacityteam.org/), Xenocanto, and eBird, volunteers can source audio data to create sound reference libraries. These libraries can then be used to train deep learning object detection models that attempt to answer the question "does this survey waveform _look_ like the species waveform?". The model can then tally up total calls and estimate population sizes (through the species average call rates)

<p align="center"><sup>Note: the tool doesn't account for imperfect detection (for example, a species at 40m may only be heard 8% of the time - which would result in undercounting). Note also that the data presented on this repository is anonymised as some species are incredibly rare, and poaching activity is well known in the area</sup></p>

![Copy of Copy of Bottom 3 (2)](https://user-images.githubusercontent.com/122735369/215263692-78bd5a51-9120-4ff6-b6d3-6f26c6bbf132.png)
<p align="center"><sup>Left: Example waveform comparison between a bird call and background noise. Right: Chart showing how the trained models compare - highlighting models that need extra training files</sup></p>

### Next Steps

The tool has been field tested with promising results, and the following steps are recommended to improve implementation:

1. Performance between the tool and manual identification methods should be compare

2. As the computer hardware on site was limited, models with small neuronal layers were utilised. Cloud software should be trialled

3. The script can be put into Streamlit for a simplified interface

4. Specific research questions with defined metrics should be planned, as insights from each survey can be used to improve future survey efforts

## What are some of the insights we can discover?

Species richness is the count of the unique species, and is an important measure of biodiversity. In the abbreviated table below we can see no change (possibly due to the island context with migration being limited to those with the capacity to cross the sea). **This metric can be used to evaluate conservation efforts, as well as providing evidence for policy lobbying.**

![Copy of Copy of Bottom 3 (3)](https://user-images.githubusercontent.com/122735369/215264531-1eb4be82-d189-430c-9fc9-6dd15f8962fa.png)

<p align="center"><sup>Right: Example view from where one audio detector was placed</sup></p>

In the following abbreviated population tables, a relationship can be seen where species of smaller group size had both lower call rates per unit time as well as fewer total calls (possibly due to a lack of 'safety in numbers'), while species with the largest maximum group sizes did were seen to not have the highest call rates (possibly due to the risk of being be drowned out). **Insights discovered here can be used for generating future research ideas or planning of future survey locations.**

![Bottom 3](https://user-images.githubusercontent.com/122735369/215263543-9525ba62-15c4-4f82-a8e2-3bbc7b28d916.png)

<p align="center"><sup>Charts showing trends in various aspects of call behaviour</sup></p>

<p align="center"><sup> This code was produced to help a wildlife charity - forks are encouraged that expand upon the script with additional features (particularly if they regard more efficient model architecture, enhanced visualisations, and/or improving work flow/process checks). If any organisations want help implementing this tool, feel free to reach out</sup></p>
