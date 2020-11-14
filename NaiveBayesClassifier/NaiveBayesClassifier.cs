using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;

namespace NaiveBayesClassifier
{
    /// <summary>
    /// Gneric Naive Bayes Classifier that works with labeled data represented
    /// as feature vectors consisting of floats.
    /// </summary>
    class NaiveBayesClassifier
    {
        private List<FeatureLikelihood> likelihood;
        private DataFeaturesContainer data;
        private List<int> targets;
        private List<int> classes;
        private Dictionary<int, double> prior;
        private int featuresVectorLength;
        private bool trained = false;

        /// <summary>
        /// Fits a Naive Bayse Classifier to the provided data and prior
        /// Prior provided must be a probability and it must be passed as a dictionary that includes all possible classes
        /// </summary>
        /// <param name="features">The data on which the model will be trained</param>
        /// <param name="targets">Target lables of the data as integer list</param>
        /// <param name="prior">The prior to be used in the model</param>
        public void Fit(DataFeaturesContainer features, List<int> targets, Dictionary<int, double> prior)
        {
            if (prior.Count != targets.Distinct().Count())
            {
                Log.Error("NaiveBayesClassifier.Fit", "Provided prior probabilities do not match with the number of classes in the provided targets");
            }

            double sum = 0;
            bool propper = true;
            foreach( var key in prior.Keys)
            {
                sum += prior[key];
                if (prior[key] > 1.0 || prior[key] < 0.0)
                    propper = false;
            }

            if (sum != 1.0 || !propper)
            {
                Log.Error("NaiveBayesClassifier.Fit", "Provided prior is not a propper probability");
            }

            this.data = features;
            this.targets = targets;
            this.prior = prior;
            this.classes = targets.Distinct().ToList();
            this.likelihood = new List<FeatureLikelihood>();
            this.featuresVectorLength = features.Shape.Columns;
            this.CalculateLikelihood();
            this.trained = true;
        }

        /// <summary>
        /// Classifies the [features] vector with the trained model.
        /// Returns the classification prediction as an integer.
        /// Model must first be [Fit] before performing predictions.
        /// If model is not [Fit] or the provided feature vector does not match the length of the training data features
        /// the function returns NULL.
        /// </summary>
        /// <param name="features">Features vector that is to be classified</param>
        /// <returns>
        /// A integer value representing the predicted class of the feature vector. 
        /// NULL if something goes wrong.
        /// </returns>
        public int? Predict(FeaturesVector features)
        {
            if (!this.trained)
                // DEV NOTE: Not best practice, since the targets provided to the classifier might contain -1 as a class
                return null;

            if (features.Count != this.featuresVectorLength)
            {
                Log.Error("NaiveBayesClassifier.Predict", "The feature vector provided is not the same shape as the training data");
                
                // DEV NOTE: Not best practice, since the targets provided to the classifier might contain -1 as a class
                return null;
            }

            Dictionary<int, double> posteriors = new Dictionary<int, double>();
            foreach (int target in this.classes)
            {
                double sum = 0;
                for (int i = 0; i < this.featuresVectorLength; i++)
                {
                    double value = features[i];
                    double p_likelihood = 0.0;
                    try
                    {
                        p_likelihood = this.likelihood[i][value][target];
                    }
                    catch (KeyNotFoundException)
                    {
                        // Skipping features that contain values that have not been seen during training
                        continue;
                    }

                    double normalization_factor = this.classes.Select(x => this.likelihood[i][value][x] * this.prior[x]).Sum();
                    double p_posterior = (p_likelihood * prior[target]) / normalization_factor;

                    // Computing combined posterior probability for all features
                    sum += Math.Log(1 - p_posterior) - Math.Log(p_posterior);
                }

                double posterior = Math.Pow(Math.E, sum);
                posterior += 1;
                posterior = 1.0 / posterior;
                posteriors[target] = posterior;
            }

            /// Implementation for ArgMax
            /// From @"https://stackoverflow.com/questions/2805703/good-way-to-get-the-key-of-the-highest-value-of-a-dictionary-in-c-sharp"
            int max = posteriors.Aggregate((l, r) => l.Value > r.Value ? l : r).Key;
            //Console.WriteLine(string.Format("spam: {0}, ham: {1}, max: {2}", posteriors[0], posteriors[1], max));
            return max;
        }

        /// <summary>
        /// Classifies features vectors and returns a list of integers that represent the predicted classes.
        /// If model is not [Fit] first returns NULL.
        /// </summary>
        /// <param name="data">Data to classify</param>
        /// <returns></returns>
        public List<int> Predict(List<FeaturesVector> data)
        {
            if (!this.trained)
                return null;

            List<int> predictions = data.Select(x => this.Predict(x) ?? -1).ToList();
            return predictions;
        }

        /// <summary>
        /// Helper function for easier code management
        /// Uses the provided data and targets to calculate the likelihoods for the classifier
        /// </summary>
        private void CalculateLikelihood()
        {
            var data_and_targets = this.data.Data.Zip(this.targets, (d, t) => new { item = d, target = t });
            Dictionary<int, int> target_counts = new Dictionary<int, int>();
            foreach (var target in this.classes)
            {
                target_counts[target] = this.targets.Where(x => x == target).Count();
            }

            for (int i = 0; i < this.featuresVectorLength; i++)
            {
                FeatureLikelihood fl = new FeatureLikelihood();
                foreach (double value in this.data.GetColumn(i).Distinct())
                {
                    Dictionary<int, double> probabilities = new Dictionary<int, double>();
                    foreach (int target in this.classes)
                    {
                        int count_per_target_and_value = data_and_targets.Where(x => x.item[i] == value && x.target == target).Count();
                        double p = (double)count_per_target_and_value / target_counts[target];
                        probabilities[target] = p;
                    }

                    fl[value] = probabilities;
                }

                this.likelihood.Add(fl);
            }
        }
    }

    /// <summary>
    /// Helper wrapper. A datastructure used to store likelithoods for a single feature.
    /// Stores probabilities for each value of the feature and for each class of the dataset.
    /// </summary>
    class FeatureLikelihood
    {
        private Dictionary<double, Dictionary<int, double>> likelihood;

        public FeatureLikelihood()
        {
            this.likelihood = new Dictionary<double, Dictionary<int, double>>();
        }
        
        public Dictionary<int, double> this[double i]
        {
            get 
            {
                return this.likelihood[i];
            }
            set { this.likelihood[i] = value; }
        }
    }

    /// <summary>
    /// Data structure used to represent the data as a List of [FeatureVector] objects.
    /// </summary>
    class DataFeaturesContainer
    {
        public List<FeaturesVector> Data { get; private set; }
        private int width;
        public int Count
        {
            get
            {
                return this.Data?.Count() ?? 0;
            }
        }

        public (int Rows, int Columns) Shape
        {
            get
            {
                return (this.Count, this.width);
            }
        }

        private DataFeaturesContainer(List<FeaturesVector> data, int vectoLength)
        {
            this.Data = data;
            this.width = vectoLength;
        }

        /// <summary>
        /// Public factory to allow user to create a [DataFeaturesContainer] objects if the parameters are valid.
        /// If the [data] parameter is NULL or empty the factory returns NULL.
        /// If any of the [FeaturesVector] objects has a different length NULL is returned.
        /// </summary>
        /// <param name="data"></param>
        /// <returns></returns>
        public static DataFeaturesContainer Create(List<FeaturesVector> data)
        {
            int vectorLength = data[0].Count;
            if(data == null || data.Count == 0)
            {
                Log.Error("DataFeaturesContainer.Create", "Feature vector list that was passed was NULL or empty");
                return null;
            }

            if(!data.All(item => item.Count == vectorLength))
            {
                Log.Error("DataFeaturesContainer.Create",  "Feature vectors are not all of the same length");
                return null;
            }

            return new DataFeaturesContainer(data, vectorLength);
        }

        /// <summary>
        /// Returns a column of the data matrix. This represents a single feature for all data-points.
        /// Returns NULL if ingex is outside of range.
        /// </summary>
        /// <param name="index"></param>
        /// <returns></returns>
        public List<double> GetColumn(int index)
        {
            if (index >= this.width || index < 0) 
            {
                Log.Error("DataFeaturesContainer.GetColumn", "The index that was passed is out of range");
                return null;
            }

            return this.Data.Select(x => x[index]).ToList();
        }

        /// <summary>
        /// Returns a row of the data matrix. This represents a single features vector or a data-point from the dataset.
        /// Returns NULL if ingex is outside of range.
        /// </summary>
        /// <param name="index"></param>
        /// <returns></returns>
        public List<double> GetRow(int index)
        {
            if (index >= this.Count || index < 0)
            {
                Log.Error("DataFeaturesContainer.GetRow", "The index that was passed is out of range");
                return null;
            }

            return this.Data[index].Features;
        }
    }

    /// <summary>
    /// Data structure that represents a features vector consisting of double values.
    /// It is a wrapper class for a [List<double>].
    /// </summary>
    class FeaturesVector
    {
        public List<double> Features { get; private set; }
        public int Count
        {
            get
            {
                return this.Features?.Count() ?? 0;
            }
        }

        public FeaturesVector(List<double> feature)
        {
            this.Features = feature;
        }

        public double this[int i]
        {
            get { return this.Features[i]; }
        }
    }
}
