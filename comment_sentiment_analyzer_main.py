from youtube_comment_downloader import *
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
from lyricsgenius import Genius



def get_comment_sentiment(downloader,analyzer,link,range):

    comments = downloader.get_comments_from_url(link)
    
    sentiment = []

    average = 0.0
    count = 0.0

    #sets up Dictionary to hold sentiments
    #Sentiment field is 0 for neutral, 1 for positive, -1 for negative
    sentiments = {'text':[],
                  'pos':[],
                  'neg':[],
                  'neu':[],
                  'compound':[],
                  'sentiment':[]}

    #iterates through the comments and assigns them a sentiment score,
    for comment in comments:
        #get the comment text
        text = comment['text']

        #puts text through analyzer and receives a score
        score = analyzer.polarity_scores(text)

        #loads sentiment scores into a dictionary
        sentiments['text'].append(text)
        sentiments['pos'].append(score['pos'])
        sentiments['neg'].append(score['neg'])
        sentiments['neu'].append(score['neu'])
        sentiments['compound'].append(score['compound'])
        compound_score = score['compound']

        print(score)

        #adds compound_score to average
        average = average + compound_score
        count = count + 1.0
        #sentiment.append([text,compound_score])

        negative_range = range * -1
        
        if compound_score >= range:
            sentiments['sentiment'].append(1)
        elif compound_score <= negative_range:
            sentiments['sentiment'].append(-1)
        else:
            sentiments['sentiment'].append(0)
        
    return pd.DataFrame(sentiments)   


def main():

    #sets up analyzer, Lexicon, and Youtbue comment downloader
    nltk.download('vader_lexicon')
    downloader = YoutubeCommentDownloader()
    analyzer = SentimentIntensityAnalyzer()
    
    #Gets user Input
    link = input('Input YouTube Link here: ')

    
    #Gets sentiment of YOutube Comments
    sentiment = get_comment_sentiment(downloader,analyzer,link)

    print(sentiment)



    
if __name__ == '__main__':
    main()