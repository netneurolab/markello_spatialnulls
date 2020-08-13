# Recommendations for spatial nulls

So you've read through the manuscript, made it this far in the walkthrough, and are finally interested in using spatial nulls in your own work.
Now, your big question is: "what null framework do you recommend?"

## Our Official Recommendation™

You'll note that we were rather careful to avoid making an explicit recommendation in our manuscript / this walkthrough because, as we tried to show in our analyses, the performance of the spatially-constrained null frameworks varies as a function of research context + question.
In some instances the nulls can be relatively consistent—in others, less so.
We encourage you to be cognizant of this in your own work and make your choice accordingly.

Alternatively—and this is something we're beginning to explore a bit more—you could opt to use some sort of consensus-based null.
That is, run some (or all!) of the null frameworks on your analysis and then report them all.
You could choose to interpret only those results consistent across some percentage (or all!) of the nulls.
We don't have any real recommendations on how to proceed with a consensus-based null at this point in time, but it's certainly something to consider in the future.

## Our Unofficial Recommendation™

With that out of the way, we can try and give you some insight into what _we intend to do_ in the future: generally speaking, we'll be using the method proposed in [Burt et al., 2020, *NeuroImage*](https://doi.org/10.1016/j.neuroimage.2020.117038).
No method is going to be perfect in all cases, but the theoretical underpinnings of the Burt-2020 method are sound and it's quite easy-to-use.
Critically, it can be applied to both surface + volumetric data alike—which will be increasingly important as the field of neuroimaging at-large continues to come to terms with the fact that the subcortex and cerebellum actually do things (surprise!).

The one main drawback of the Burt-2020 is its speed—it's got a bit of a computational cost, especially if you're working with multivariate data (think NeuroSynth or the Allen Human Brain Atlas) since you need to generate surrogates for each brain map independently.
That said, with computing power because more and more cost-effective, this concern really shouldn't factor in to most neuroimaging workflows (especially since running e.g., fMRIPrep on a single subject nowadays can take upwards of 10 hours!).

If you're interested in using this method in your own research head on over to [BrainSMASH](https://brainsmash.readthedocs.io/) to get started.
