# Recommendations for spatial nulls

So you've read through the manuscript, made it this far in the walkthrough, and are convinced of the need to use spatial nulls in your own work.
Now, your big question is: "what null framework do you recommend?"

We tried to make this relatively explicit in our set of recommendations and considerations at the end of the manuscript (if you haven't checked those out, please do so!), but, to put it here plainly: when working with **parcellated data**, the null framework that seems to consistently yield statistical estimates with the lowest error rate is the *Cornblath* method.
(Although, note that most of the spatial permutation models, including the *Vázquez-Rodríguez*, *Baum*, and *Cornblath*, yield remarkably similar error rates and statistical inferences across the analyses examined in our manuscript.)
When working with **vertex-level data**, the optimal null method appears to be the framework proposed by [Alexander-Bloch et al., 2018, *NeuroImage*](https://doi.org/10.1016/j.neuroimage.2018.05.070) (which we refer to as the *Vázquez-Rodríguez* method in our manuscript, for consistency with parcellated results).

However, if you have volumetric, subcortical, cerebellar, or region-of-interest data, then the aforementioned null frameworks will not work—you can only use the parameterized data models in these situations.
Here, we find that the *Moran* and *Burt-2020* methods seem to fare rather comparably.
Note though, that when working with high-resolution, volumetric data, we have found the only realistic option is to use the *Burt-2020* method; although the *Moran* method can hypothetically handle such datasets, in our experience the required eigendecomposition is computationally intractable for high-resolution data.

Alternatively—and this is something we're beginning to explore a bit more—you could opt to use some sort of consensus-based null.
That is, run some (or all!) of the null frameworks on your analysis and then report them all.
You could choose to interpret only those results consistent across some percentage of the nulls.
We don't have any real recommendations on how to proceed with a consensus-based null at this point in time, but it's certainly something to consider in the future.
