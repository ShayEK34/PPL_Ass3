import string
import nltk
import pandas as pd
import re
import matplotlib.pyplot as plt
import itertools
from nltk.stem.porter import *
from nltk.corpus import stopwords
import numpy as np
from sklearn import metrics, model_selection
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn import metrics
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


class sentiment_analysis:
    """
    This class represent sentiment analysis for twitter tweets for negative and positive tweets.
    The class contain preprocessing methods for the tweets, methods for 3 classification models, plotting methods
    and submission method for kaggle.
    """

    def __init__(self, path_train, path_test):
        """
        This constructor get the paths for the train and test csv read the files, insert the data into data frames-
        for train tweets and sentiment label and for test the tweets and the ID.
        we save the tweets labels for plotting methods and the best score for the chosen model.
        :param path_train:
        :param path_test:
        """
        df_train = pd.read_csv(path_train, encoding='latin-1')
        df_test = pd.read_csv(path_test, encoding='latin-1')
        df_train = df_train.drop_duplicates()
        df_test = df_test.drop_duplicates()
        self.tweets_train = df_train['SentimentText']
        self.sentiment_train = df_train['Sentiment']
        self.tweets_test = df_test['SentimentText']
        self.id_test = df_test['ID']
        self.labels_classes = [0, 1]
        self.best_model_cv_acc = 0
        self.lgr_chosen = False
        self.nb_chosen = False
        self.dt_chosen = False


    def preprocess(self, tweet):
        """
        This method get a single tweet and make a preprocess that include: cleaning the tweet from emojis, hashtags and more,
        removing stopwords and lemattization for the words. in the end of the process we get a single tweet after this whole
        process.
        :param tweet: The given original tweet
        :return: A clean tweet after preprocess
        """
        def clean_tweet_to_tokens(tweet):
            """
            This method clean the tweet from emojis, hashtags, urls user names and more and return a list that contain
            the words as tokens after cleaning.
            :param tweet: A original tweet.
            :return: A list of tokens after cleaning.
            """
            # Remove user @ references and '#' from tweet
            tweet = re.sub(r"http\S+|www\S+|https\S+", '', tweet, flags=re.MULTILINE)
            tweet = re.sub('\[.*?\]', '', tweet)
            tweet = re.sub('<.*?>+', '', tweet)
            tweet = re.sub('[%s]' % re.escape(string.punctuation), '', tweet)
            tweet = re.sub('\n', '', tweet)
            tweet = re.sub('\w*\d\w*', '', tweet)
            tweet = re.sub(r'\@\w+|\#', '', tweet)
            tweet = re.sub(r'[^\x00-\x7f]*', r'', tweet)

            tweet_tokens = tokens_re.findall(tweet)
            tweet_tokens = [token if emoticon_re.search(token) else token.lower() for token in tweet_tokens]
            return tweet_tokens

        def remove_stop_words(tweet):
            """
            This method get a list of tokens and removing all the stopwords from the list.
            :param tweet: The list of tokens of the tweet
            :return: A list if tokens of tweet without stopwords.
            """
            stop_words = set(stopwords.words('english'))
            for word in [".", ",", "(", ")", "<", ">", "br", "!", "/", "--", "n't", "'s", "''", "?", "...", "``", ":",
                         "-", "'", "would", ";", "*", "@", "&", "\\", "~", ";", ";)", "[", "]"]:
                stop_words.add(word)
            filtered_tweet = [w for w in tweet if not w in stop_words]
            return filtered_tweet

        def Lemmatization(tweet_without_stopwords):
            """
            This method get a list of tokens without stopwords and doing lemmatization for the the tokens and connect
            the tokens to single sentence.
            :param tweet_without_stopwords: List of tokens without stopwords.
            :return: A complete tweet without stopwords and after lemmatization.
            """
            # using porter stemmer
            # stemmer = PorterStemmer()
            # tweet = ' '.join([stemmer.stem(w) for w in tweet_without_stopwords])

            # using lemmatization
            lemmatizer = WordNetLemmatizer()
            tweet = ' '.join([lemmatizer.lemmatize(w) for w in tweet_without_stopwords])

            return tweet

        emoticons_str = r"""
            (?:
                [:=;] # Eyes
                [oO\-]? # Nose (optional)
                [D\)\]\(\]/\\OpP] # Mouth
            )"""

        regex_str = [
            emoticons_str,
            r'<[^>]+>',  # HTML tags
            r'(?:@[\w_]+)',  # @-mentions
            r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)",  # hash-tags
            r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&amp;+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+',  # URLs
            # r'(?:(?:\d+,?)+(?:\.?\d+)?)', # numbers
            # r"(?:[a-z][a-z'\-_]+[a-z])", # words with - and '
            r'(?:[\w_]+)',  # other words
            r'(?:\S)'  # anything else
        ]

        tokens_re = re.compile(r'(' + '|'.join(regex_str) + ')', re.VERBOSE | re.IGNORECASE)
        emoticon_re = re.compile(r'^' + emoticons_str + '$', re.VERBOSE | re.IGNORECASE)

        tweets_tokens = clean_tweet_to_tokens(tweet)
        # tweets_tokens_without_stop_words= remove_stop_words(tweets_tokens)
        final_tweets_tokens = Lemmatization(tweets_tokens)
        return final_tweets_tokens

    def split_data_train(self):
        """
        This method split the data train for 80% for train tweets and 20% for validation.
        :return:
        """
        self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(self.tweets_train, self.sentiment_train, test_size=0.1)


    def chosen_model(self):
        """
        This method choose the best model by the highest average accuracy for 10 cross validation, fitting the model,
        making prediction on the validation data, plot confusion matrix and print classification report that contain
        the metrics results for the chosen model:accuracy, precision, recall and F1-score.
        :return:
        """
        def plot_confusion_matrix(cm, classes,
                                  normalize=False,
                                  title='Confusion matrix',
                                  cmap=plt.cm.Blues):
            """
            This function prints and plots the confusion matrix.
            Normalization can be applied by setting `normalize=True`.
            """
            if normalize:
                cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                print("Normalized confusion matrix")
            else:
                print('Confusion matrix, without normalization')

            print(cm)

            plt.imshow(cm, interpolation='nearest', cmap=cmap)
            plt.title(title)
            plt.colorbar()
            tick_marks = np.arange(len(classes))
            plt.xticks(tick_marks, classes, rotation=45)
            plt.yticks(tick_marks, classes)

            fmt = '.2f' if normalize else 'd'
            thresh = cm.max() / 2.
            for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
                plt.text(j, i, format(cm[i, j], fmt),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")

            plt.ylabel('True label')
            plt.xlabel('Predicted label')
            plt.tight_layout()
            plt.show()

        if(self.lgr_chosen):
            print("The best model that chosen with the highest avg accuracy of CV is: Logistic Regression!")
            print("...fit model...")
            self.pipeLine_LogicReg.fit(self.x_train, self.y_train)
            self.predictions_val = self.pipeLine_LogicReg.predict(self.x_val)
            print('Accuracy:', metrics.accuracy_score(self.y_val, self.predictions_val))
            print('Classification Report:')
            print(classification_report(self.y_val, self.predictions_val))
            print('Confusion Matrix:')
            cnf_matrix = confusion_matrix(self.y_val, self.predictions_val)
            np.set_printoptions(precision=2)
            # Plot confusion matrix
            plt.figure()
            plot_confusion_matrix(cnf_matrix, classes=self.labels_classes,
                                  title='Logistic Regression - Confusion matrix')

        if (self.nb_chosen):
            print("The best model that chosen with the highest avg accuracy of CV is: Naive Bayes!")
            print("...fit model...")
            self.pipeLine_Naive_Bayes.fit(self.x_train, self.y_train)
            self.predictions_val = self.pipeLine_Naive_Bayes.predict(self.x_val)
            print('Accuracy:', metrics.accuracy_score(self.y_val, self.predictions_val))
            print('Classification Report:')
            print(classification_report(self.y_val, self.predictions_val))
            print('Confusion Matrix:')
            cnf_matrix = confusion_matrix(self.y_val, self.predictions_val)
            np.set_printoptions(precision=2)
            # Plot confusion matrix
            plt.figure()
            plot_confusion_matrix(cnf_matrix, classes=self.labels_classes,
                                  title='Naive Bayes - Confusion matrix')


    def logistic_regression_model(self):
        """
        This method create pipe line for logistic regression model. The pipe contain three steps:
        1. making preprocess on the tweets and then doing count vector for the words.
        2. creating TF/IDF for the result of the count vector.
        3. activating the logistic regression model on the data after stages 1,2.
        The we doing 10 cross validation on the data using the defined pipe line and plotting
        the average metrics for accuracy, precision and recall.

        :return:
        """
        print("Logistic Regression model:")

        print("...Creation of pipeline-Logistic Regression...")

        self.pipeLine_LogicReg = Pipeline([
            ('data', CountVectorizer(binary=True, ngram_range=(1, 2), preprocessor=self.preprocess)),
            ('tfidf', TfidfTransformer(use_idf=True, norm='l2')),
            ('clf', LogisticRegression(C=1.0, penalty='l2', solver='liblinear')),
        ])
        print("...start cross validation (with preprocceing)...")
        # 10-Cross-validation
        scores_lgr = model_selection.cross_validate(self.pipeLine_LogicReg, self.x_train, self.y_train, cv=10,
                                      scoring=["accuracy", "precision", "recall"])

        self.accuracy_mean_lgr =scores_lgr["test_accuracy"].mean()

        print("----------------Logistic Regression metrics for 10 CV on train: -------------------")
        print("The average accuracy for 10 CV is: ", str(round(scores_lgr["test_accuracy"].mean()* 100,3)),
        "\nThe average precision for 10 CV is: ",str(round(scores_lgr["test_precision"].mean()* 100,3)),
        "\nThe average recall for 10 CV is: ", str(round(scores_lgr["test_recall"].mean()* 100,3)))
        print("-------------------------------------------------------------------------")

        if(self.accuracy_mean_lgr > self.best_model_cv_acc ):
            self.best_model_cv_acc = self.accuracy_mean_lgr
            self.lgr_chosen = True
            self.nb_chosen = False
            self.dt_chosen = False

    def naive_bayes_model(self):
        """
        This method create pipe line for naive bayes model. The pipe contain three steps:
        1. making preprocess on the tweets and then doing count vector for the words.
        2. creating TF/IDF for the result of the count vector.
        3. activating the naive bayes model on the data after stages 1,2.
        The we doing 10 cross validation on the data using the defined pipe line and plotting
        the average metrics for accuracy, precision and recall.
        :return:
        """
        print("Naive Bayes model:")

        print("...Creation of pipeline-Naive Bayes...")

        self.pipeLine_Naive_Bayes = Pipeline([
            ('data', CountVectorizer(binary=True, ngram_range=(1, 2),preprocessor=self.preprocess)),  # strings to token integer counts
            ('tfidf', TfidfTransformer(use_idf=True, norm='l2')),  # integer counts to weighted TF-IDF scores
            ('clf', MultinomialNB()),
        ])
        print("...start cross validation (with preprocceing)...")
        # 10-Cross-validation

        scores_nb = model_selection.cross_validate(self.pipeLine_Naive_Bayes, self.x_train, self.y_train, cv=10,
                                                    scoring=["accuracy", "precision", "recall"])

        self.accuracy_mean_nb = scores_nb["test_accuracy"].mean()

        print("----------------Naive Bayes metrics for 10 CV on train: -------------------")
        print("The average accuracy for 10 CV is: ", str(round(scores_nb["test_accuracy"].mean() * 100, 3)),
              "\nThe average precision for 10 CV is: ", str(round(scores_nb["test_precision"].mean() * 100, 3)),
              "\nThe average recall for 10 CV is: ", str(round(scores_nb["test_recall"].mean() * 100, 3)))
        print("-------------------------------------------------------------------------")

        if (self.accuracy_mean_nb > self.best_model_cv_acc):
            self.best_model_cv_acc = self.accuracy_mean_nb
            self.lgr_chosen = False
            self.nb_chosen = True
            self.dt_chosen = False


    def decisionTree_model(self):
        """
        This method create pipe line for decision tree model. The pipe contain three steps:
        1. making preprocess on the tweets and then doing count vector for the words.
        2. creating TF/IDF for the result of the count vector.
        3. activating the decision tree model on the data after stages 1,2.
        The we doing 10 cross validation on the data using the defined pipe line and plotting
        the average metrics for accuracy, precision and recall.
        :return:
        """
        print("Decision Tree model:")

        print("...Creation of pipeline-Decision Tree...")

        self.pipeLine_dt = Pipeline([
            ('data', CountVectorizer(binary=True, ngram_range=(1, 2),preprocessor=self.preprocess)),  # strings to token integer counts
            ('tfidf', TfidfTransformer(use_idf=True, norm='l2')),  # integer counts to weighted TF-IDF scores
            ('clf', DecisionTreeClassifier(random_state=0)),
        ])
        print("...start cross validation (with preprocceing)...")
        # 10-Cross-validation

        scores_nb = model_selection.cross_validate(self.pipeLine_dt, self.x_train, self.y_train, cv=10,
                                                    scoring=["accuracy", "precision", "recall"])

        self.accuracy_mean_dt = scores_nb["test_accuracy"].mean()

        print("----------------Decision Tree metrics for 10 CV on train: -------------------")
        print("The average accuracy for 10 CV is: ", str(round(scores_nb["test_accuracy"].mean() * 100, 3)),
              "\nThe average precision for 10 CV is: ", str(round(scores_nb["test_precision"].mean() * 100, 3)),
              "\nThe average recall for 10 CV is: ", str(round(scores_nb["test_recall"].mean() * 100, 3)))
        print("-------------------------------------------------------------------------")

        if (self.accuracy_mean_dt > self.best_model_cv_acc):
            self.best_model_cv_acc = self.accuracy_mean_dt
            self.lgr_chosen = False
            self.nb_chosen = False
            self.dt_chosen = True

    def plot_cv_results(self):
        """
        This method plotting the average accuracy results of the 10 cross validation of each model.
        :return:
        """
        objects = ('Logisic Regression', 'Naive Bayes','Decision Tree')
        y_pos = np.arange(len(objects))
        performance = [self.accuracy_mean_lgr*100,self.accuracy_mean_nb*100, self.accuracy_mean_dt*100]
        plt.bar(y_pos, performance, align='center', alpha=0.2)
        plt.xticks(y_pos, objects)
        plt.ylabel('Accuracy')
        plt.title('Models training CV average accuracy results:')
        plt.show()

    def to_csv_submission(self):
        """
        This method use the best chosen model with the highest average accuracy of 10 cross validation and
        make prediction in the test tweets data, creating data frame with two columns: ID- the id of tweet i test file,
        Sentiment- the predicted sentiment of tweet in test. after creating the data framme we extract this data to
        csv file for kaggle submission.
        :return:
        """
        pred = self.pipeLine_LogicReg.predict(self.tweets_test)
        df_predictions = pd.DataFrame(pred)
        df_id = pd.DataFrame(self.id_test)
        df_id["Sentiment"] = df_predictions
        df_id.to_csv('submission_file.csv', index=False)


if __name__ == '__main__':
    path_train = r"Train.csv"
    path_test = r"Test.csv"
    s_analysis = sentiment_analysis(path_train, path_test)
    s_analysis.split_data_train()
    s_analysis.logistic_regression_model()
    s_analysis.naive_bayes_model()
    s_analysis.decisionTree_model()
    s_analysis.plot_cv_results()
    s_analysis.chosen_model()
    s_analysis.to_csv_submission()
