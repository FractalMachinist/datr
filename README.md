# Datr

Based on [this OkCupid dataset](https://www.kaggle.com/datasets/andrewmvd/okcupid-profiles), fine tune a tiny LLM to respond to dating profile essay questions conditioned on the rest of the profile. Set up a local web app which allows a user to specify a hypothetical profile's non-essay features and generate essay responses.

CURL:

```bash
curl -L -o ./downloads/okcupid-profiles.zip\
  https://www.kaggle.com/api/v1/datasets/download/andrewmvd/okcupid-profiles
```

(We don't need kaggle credentials for this)

## Architecture Decisions

While this *could* be done by simply prepending a text version of a profile to the start of the prompt and fine-tuning, I want to (for example) take the derivative of the essay with respect to height. That means all the traits need to exist as vectors, and those vectors need grafted into some part of the LLM's processing.

This means traits like:
- Continuous values like height and income map to scalars
  - Unspecified values are replaced with a random draw from the dataset
- Categorical values like education and ethnicity map to one-hot vectors
  - Unspecified values are mapped to zero-hot vectors
- Location is omitted
- Ordered categorical values like "drugs"  (never, sometimes, often) are mapped to ranges of scalars partitioning 0 to 1
  - Again, unspecified values are replaced with a random draw from the dataset
- Multi-categorical values ("Whether or not the person has kids or plans of having them") are split into individual categorical values. This applies to "pets" and "offspring" and "ethnicity".

Because we're tinkering with the actual architecture, we expect we won't be able to use existing off the shelf LLM fine-tuning libraries. That's fine! We're fluent in PyTorch and Tensorflow; we'll crack it open and get it to work. I'd like to practice pytorch.

For ease, we'll run the literal smallest model we can find. Let's use https://huggingface.co/arnir0/Tiny-LLM

## Data Survey

**Traits:**
status
sex
orientation
body_type
diet
drinks
drugs
education
ethnicity
height
income
job
last_online
location
offspring
pets
religion
sign
smokes
speaks

**Essays:**
essay0- My self summary
essay1- What I’m doing with my life
essay2- I’m really good at
essay3- The first thing people usually notice about me
essay4- Favorite books, movies, show, music, and food
essay5- The six things I could never do without
essay6- I spend a lot of time thinking about
essay7- On a typical Friday night I am
essay8- The most private thing I am willing to admit
essay9- You should message me if...

# Development Stages
## Stage 1: Get Data & Build Data Pipeline

- [ ] Download the dataset and investigate it to see what it's about
- [ ] Refine our trait vector scheme
  - Write a Python function to convert a profile's traits (tolerating missing entries!) into a NumPy array
- [ ] Write a loader for our dataset
  - Structure:
    - Input: the traits vector of a profile
    - Output: Text like "My self summary\n{the self summary from that profile}"
  - This means each profile's trait vector will appear in the dataset once per essay that profile provided.

## Stage 2: Fancy Fine-Tuning

- [ ] Decide on a *tiny* model for our first attempt (probably something from HuggingFace)
- [ ] Figure out how we can open up the model's actual content (e.g. its various attention layers and perceptron layers) so we can splice in our trait vector
- [ ] Decide where in the model to splice in our trait vector
  - Early?
  - Middle?
  - Late?
- [ ] Implement a method to condition the model on traits and run its output (the function we'll train).
- [ ] Demonstrate running the model with spliced trait vector
  - Note: our splicing should be initialized with 0's weights so it doesn't alter existing behavior.
- [ ] Execute the training & keep logs

## Stage 3: Tinker!

At this point, we'd like to be able to specify a traits vector and an essay prompt, then get an answer.