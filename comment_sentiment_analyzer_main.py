from youtube_comment_downloader import *
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd



def get_comment_sentiment(downloader,analyzer,link):

    comments = downloader.get_comments_from_url(link)
    
    sentiment = []

    average = 0.0
    count = 0.0
    positive_count= 0
    negative_count = 0
    neutral_count = 0

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


        
        if compound_score >= 0.05:
            sentiments['sentiment'].append(1)
        elif compound_score <= -0.05:
            sentiments['sentiment'].append(-1)
        else:
            sentiments['sentiment'].append(0)
        
    return pd.DataFrame(sentiments)   


def main():
    nltk.download('vader_lexicon')
    downloader = YoutubeCommentDownloader()
    link = input('Input YouTube Link here: ')
    analyzer = SentimentIntensityAnalyzer()
    sentiment = get_comment_sentiment(downloader,analyzer,link)

    print(sentiment)



    
if __name__ == '__main__':
    main()