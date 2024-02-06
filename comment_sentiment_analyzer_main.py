from youtube_comment_downloader import *
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer




def main():
    nltk.download('vader_lexicon')
    downloader = YoutubeCommentDownloader()
    comments = downloader.get_comments_from_url('ENTER YOUTUBE LINK HERE')
    analyzer = SentimentIntensityAnalyzer()
    for comment in comments:
        print(comment['text'])
        print(analyzer.polarity_scores(comment['text']))

    
if __name__ == '__main__':
    main()