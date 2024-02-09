from youtube_comment_downloader import *
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd



def get_comment_sentiment(downloader,analyzer,link,range):

    comments = downloader.get_comments_from_url(link)
    
    sentiment = []

    average = 0.0
    count = 0.0

    sentiments = {'text':[],
                  'pos':[],
                  'neg':[],
                  'neu':[],
                  'compound':[],
                  'sentiment':[]}

    for comment in comments:
        
        text = comment['text']
        score = analyzer.polarity_scores(text)

        sentiments['text'].append(text)
        sentiments['pos'].append(score['pos'])
        sentiments['neg'].append(score['neg'])
        sentiments['neu'].append(score['neu'])
        sentiments['compound'].append(score['compound'])


        print(score)
        compound_score = score['compound']
        average = average + compound_score
        count = count + 1.0
        sentiment.append([text,compound_score])

        negative_range = range * -1
        
        if compound_score >= range:
            sentiments['sentiment'].append(1)
        elif compound_score <= negative_range:
            sentiments['sentiment'].append(-1)
        else:
            sentiments['sentiment'].append(0)
        
    return pd.DataFrame(sentiments)   


def main():
    nltk.download('vader_lexicon')
    downloader = YoutubeCommentDownloader()
    link = input('Input YouTube Link here: ')
    range = input('Enter Sentiment Threshold: ')

    analyzer = SentimentIntensityAnalyzer()
    sentiment = get_comment_sentiment(downloader,analyzer,link,range)

    print(sentiment)



    
if __name__ == '__main__':
    main()