from youtube_comment_downloader import *
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer



def get_comment_sentiment(downloader,analyzer,link):

    comments = downloader.get_comments_from_url(link)
    
    sentiment = []

    average = 0.0
    count = 0.0
    for comment in comments:
        score = analyzer.polarity_scores(comment['text'])
        average = average + score['compound']
        count = count + 1.0
        sentiment.append([comment['text'],score['compound']])
    average = average/count
    sentiment.append(average)
    return sentiment    


def main():
    nltk.download('vader_lexicon')
    downloader = YoutubeCommentDownloader()
    link = input('Input YouTube Link here: ')
    analyzer = SentimentIntensityAnalyzer()
    sentiment = get_comment_sentiment(downloader,analyzer,link)

    for score in sentiment[:-1]:
        print('Comment: ' + score[0])
        print('Score: ' + str(score[1])+'\n\n\n')

    print("Average Score: " + str(sentiment[-1]))    



    
if __name__ == '__main__':
    main()