import os
from openai import OpenAI
from pydantic import BaseModel
import instructor
from typing import Literal
import pandas as pd
from tqdm import tqdm

os.environ['OPENAI_API_KEY'] = 'sk-proj-fZ0EZndm31kDPfP3kw9IT3BlbkFJ1oxnK7eEJVnldV8joJ5i'

client = instructor.from_openai(OpenAI())

class AdOrNot(BaseModel):
    ad_or_not: int
    confidence: Literal['high', 'medium', 'low']
    reasoning: str

def get_analysis(cc_transcript: str) -> AdOrNot:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": cc_transcript},
    ]
    response = client.chat.completions.create(
        model="gpt-4o-2024-08-06",
        messages=messages,
        response_model=AdOrNot,
    )
    return response

system_prompt = """
You are a ad classifier. You are given a block of text and you need to determine if the text is an ad or not. 
You need to consider the following factors when making your decision:

1. Entire Text Focus: Analyze the entire block of text provided, paying attention to the overall context. Ads and non-ads may coexist within the same text block.
2. Span Tag Emphasis: if sentence contain html tag, focus specifically on sentences enclosed within span html tags, as these are critical for the classification task.
3. multiple topics: the text may contain ads and non-ads. In such cases, focus on the sentence with html tages first and determine if it is an ad or not.
4. Calls to Action and promotional languages: Look for calls to action, such as "buy now," "click here," or "visit our website," which are common in ads. However, be cautious, as these may also appear in non-ad content.
5. Ambiguity and Judgment: In ambiguous cases, apply your best judgment and adjust the confidence level accordingly. Consider the context and the entirety of the text.
6. Tone and Style: Consider the tone, style, and formality of the text, as these may provide additional clues for classification.
7. genre: New, TV shows, and speeches may contain calls to action but are not ads. Be cautious when classifying these types of content.
8. first person pronouns: the use of first person pronouns like "I" and "we" may be speach or news articles, not ads.
9. Non-Ad Response: If the text is clearly not an ad, respond with:
'ad_or_not': 0
'confidence': 'high'
'reasoning': the actual reasoning
10. Ad Response: If the text is clearly an ad, respond with:
'ad_or_not': 1
'confidence': 'high'
'reasoning': the actual reasoning
11. Language and Format: Ads are generally more likely to have correct punctuation, spelling, and capitalization. However, these attributes are not definitive indicators.

examples of classification:
"there is nothing more important in the hispanic community than family. this virus pulls at the very fabric of who we are, preventing us from doing what we love most... 
to come to the table and celebrate life and family. this pandemic isnt over, so please, get the vaccine. and if youve already been vaccinated, get your booster. its free, its safe, and it can be the difference between life and death. lets come together and unite to prevent.
Please fill out the pydantic model with this information." This is a non-ad content, the reasoning is that the text is a public service announcement about the importance of getting vaccinated.

"get your free trial today! click here to sign up now!" This is an ad content, the reasoning is that the text contains a call to action to sign up for a free trial.
"""

df = pd.read_csv("trial_200 .csv")

ads = df['cc_text'].tolist()

objects = []

for ad in tqdm(ads, desc="Analyzing ads"):
    objects.append(get_analysis(ad))

df['ad_or_not'] = [o.ad_or_not for o in objects]
df['confidence'] = [o.confidence for o in objects]
df['reasoning'] = [o.reasoning for o in objects]

df.to_csv("ad_200_with_analysis.csv", index=False)