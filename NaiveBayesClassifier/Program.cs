using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;

namespace NaiveBayesClassifier
{
    class Program
    {
        static void Main(string[] args)
        {
            // Reading data from .txt files
            List<_DataSetItem> raw_data_train = ReadInData(Properties.Resources.SMSSpamTrain);
            List<_DataSetItem> raw_data_test = ReadInData(Properties.Resources.SMSSpamTest);
            List<_DataSetItem> full_dataset = new List<_DataSetItem>(raw_data_test.Count + raw_data_train.Count);
            full_dataset.AddRange(raw_data_train);
            full_dataset.AddRange(raw_data_test);

            // Extracting the vocabulary of the dataset. Only words that occur 3 or more times are considered part of the vocabulary
            HashSet<string> vocabulary = ExtractVocabulary(full_dataset, 3);

            // Transforming the raw SMS data into a bag of words from the previously extracted vocabulary
            DataFeaturesContainer X_train = BagOfWordsFromVocabularyAndSMS(raw_data_train, vocabulary);
            DataFeaturesContainer X_test = BagOfWordsFromVocabularyAndSMS(raw_data_test, vocabulary);

            // Assigning the target values
            List<int> Y_train = raw_data_train.Select(x => x.target == "spam" ? 0 : 1).ToList();
            List<int> Y_test = raw_data_test.Select(x => x.target == "spam" ? 0 : 1).ToList();

            // Defining the prior
            // Unbiased prior, carries no information, every class is as likely to occur
            Dictionary<int, double> unbiased_prior = new Dictionary<int, double>()
            { 
                { 1, 0.5 }, 
                { 0, 0.5 } 
            };

            // Reasearched prior found on the wikipedia articke
            // @"https://en.wikipedia.org/wiki/Naive_Bayes_spam_filtering"
            Dictionary<int, double> wiki_prior = new Dictionary<int, double>()
            {
                { 1, 0.2 },
                { 0, 0.8 }
            };

            // Calculating the prior from our dataset which includes finding the frequency of 
            // actuall spam messages vs ham messages
            List<string> tartets = full_dataset.Select(x => x.target).ToList();
            double spam_prior = CalculateDatasetPrior(tartets);
            Dictionary<int, double> dataset_calculated_prior = new Dictionary<int, double>()
            {
                { 1, 1.0 - spam_prior },
                { 0, spam_prior }
            };

            // Training the model and doing the predictions
            NaiveBayesClassifier model = new NaiveBayesClassifier();
            model.Fit(X_train, Y_train, unbiased_prior);
            List<int> predictions = model.Predict(X_test.Data);

            // Measuring model performance
            ModelMetrics modelMetrics = new ModelMetrics(Y_test, predictions, 0);
            modelMetrics.PrintMeasures();

            // Saving the results 
            List<string> test_sms = raw_data_train.Select(x => x.sms.Trim()).ToList();
            WriteResults("results.tsv", test_sms, predictions);
        }

        /// <summary>
        /// Used for calculating a prior for the SMS dataset. It is just a helper function so it is not
        /// intendet to be used in general cases.
        /// </summary>
        /// <param name="targets">List of string targets to be converted into List of integers for the classifier</param>
        /// <returns></returns>
        static double CalculateDatasetPrior(List<string> targets)
        {
            int spam_count = targets.Where(x => x == "spam").Count();
            return (double)spam_count / targets.Count;
        }
        
        /// <summary>
        /// Function for converting a lentence dataset into a Bag of Words representation to be used for model treining.
        /// The text segment (sms) of each data item is split into words. Separation by whitespace and symbols is used.
        /// Only words consisting of alphanumeric values are taken into account from each text item.
        /// Returns a [DataFeaturesContainer] containing the bag of words as a the [Data] field.
        /// </summary>
        /// <param name="raw_data">Data to be processed</param>
        /// <param name="vocabulary">The vocabulary used for Bag of Words representation</param>
        /// <returns></returns>
        static DataFeaturesContainer BagOfWordsFromVocabularyAndSMS(List<_DataSetItem> raw_data, HashSet<string> vocabulary)
        {
            List<FeaturesVector> data = new List<FeaturesVector>();
            foreach (var item in raw_data)
            {
                List<string> sms_vocab = new List<string>(Regex.Split(item.sms, @"[^a-zA-Z0-9]+"));
                List<double> tmp = vocabulary.Select(x => sms_vocab.Contains(x) ? 1.0 : 0.0).ToList();
                FeaturesVector features = new FeaturesVector(tmp);
                data.Add(features);
            }

            DataFeaturesContainer result = DataFeaturesContainer.Create(data);
            return result;
        }

        /// <summary>
        ///  Function specific for reading the "SMS Spam Collection Data Set" provided in the homework.
        ///  Should not be used as a generic function since it is dataset specific and used as a helper function
        ///  for code separation.
        /// </summary>
        /// <param name="fileContents">Contents of the file to be processed.</param>
        static List<_DataSetItem> ReadInData(string fileContents)
        {
            string[] lines = fileContents.Split('\n');
            List<_DataSetItem> data = new List<_DataSetItem>();
            foreach (string line in lines)
            {
                if (string.IsNullOrEmpty(line))
                    continue;

                string[] segments = line.Split('\t');
                var item = new _DataSetItem(segments[1], segments[0]);
                data.Add(item);
            }

            return data;
        }

        /// <summary>
        /// Writes the results from the test in a .tsv file
        /// The format is [sms_text]\t[prediction]\r\n
        /// </summary>
        /// <param name="fileName">File that will be created in the execution directory where the results are saved</param>
        /// <param name="sms">The list of SMS messages that were in the test set</param>
        /// <param name="predictions">The list of predictions made by the model for the [sms] parameter SMS messages</param>
        static void WriteResults(string fileName, List<string> sms, List<int> predictions)
        {
            var items = sms.Zip(predictions, (s, p) => new { sms = s, pred = p } );
            StringBuilder csv = new StringBuilder();
            foreach (var item in items)
            {
                string line = string.Format("{0}\t{1}", item.sms, item.pred);
                csv.AppendLine(line);
            }

            System.IO.File.WriteAllText(fileName, csv.ToString());
        }

        /// <summary>
        /// Extracts the vocabulary out of a dataset of texts.
        /// The text (sms) of each data item is split into words. Separation by whitespace and symbols is used.
        /// Only words consisting of alphanumeric values are taken into account from each text item.
        /// Words with length smaller than 2 are not considered in the vocabulary.
        /// This is done in order to remove remains of shortened negative constructs and articles (a, an)
        /// Ex: "don't", "shouldn't" ... => "don" and "t", "shouldn" and "t", we want to awoid saving "t" as a word.
        /// 
        /// Words are counted and words that occur less than [cutoffValue] are not considered as part of the vocabulary.
        /// </summary>
        /// <param name="data">Data to be processed</param>
        /// <param name="cutoffValue">Words that occur less than this value are not considered part of the vocabulary</param>
        /// <returns></returns>
        static HashSet<string> ExtractVocabulary(List<_DataSetItem> data, int cutoffValue = 0)
        {
            Dictionary<string, int> wordCounts = new Dictionary<string, int>();
            foreach (var item in data)
            {
                foreach (string word in Regex.Split(item.sms, @"[^a-zA-Z0-9]+"))
                {
                    // Words that are one character are not considered in the vocabulary
                    if (word.Length <= 1)
                        continue;

                    string filtered_word = word.ToLower();
                    if (wordCounts.ContainsKey(filtered_word))
                    {
                        wordCounts[filtered_word]++;
                    }
                    else
                    {
                        wordCounts[filtered_word] = 0;
                    }
                }
            }

            HashSet<string> vocabulary = wordCounts.Where(x => x.Value >= cutoffValue).Select(x => x.Key).ToHashSet();
            return vocabulary;
        }

        /// <summary>
        /// Helper wrapper class for easier code management
        /// Specific for the SMS dataset (sort of).
        /// </summary>
        class _DataSetItem {
            public string sms;
            public string target;

            public _DataSetItem(string sms, string target)
            {
                this.sms = sms;
                this.target = target;
            }
        }

    }
}
