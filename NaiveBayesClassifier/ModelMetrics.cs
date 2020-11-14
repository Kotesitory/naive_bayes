using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NaiveBayesClassifier
{
    /// <summary>
    /// Class that calculates model metrics from list of true values and predicted values
    /// Included metrics:
    /// - Model Accuracy
    /// - Model Precision
    /// - Model Specificity
    /// - Model F1_score
    /// - Model Confusion matrix
    /// </summary>
    class ModelMetrics
    {
        private float truePositives = 0;
        private float trueNegatives = 0;
        private float falsePositives = 0;
        private float falseNegatives = 0;

        /// <summary>
        /// The [predictions] list and [targets] list must be of same length
        /// </summary>
        /// <param name="targets">Dataset targets from test set</param>
        /// <param name="predictions">Predictions from the model</param>
        /// <param name="positive">Value representing the positive class in the model</param>
        public ModelMetrics(List<int> targets, List<int> predictions, int positive = 1)
        {
            if (targets.Count != predictions.Count)
            {
                Log.Error("ModelMetrics", "The provided targets and predictions are not of the same length");
            }

            var zipped_list = targets.Zip(predictions, (t, p) => new { trueValue = t, predictedValue = p });
            foreach (var item in zipped_list)
            {
                if(item.trueValue == item.predictedValue)
                {
                    if (item.predictedValue == positive)
                        this.truePositives++;
                    else
                        this.trueNegatives++;
                }
                else
                {
                    if (item.predictedValue == positive)
                        this.falsePositives++;
                    else
                        this.falseNegatives++;
                }
            }
        }

        /// <summary>
        /// Calculates and returns model Accuracy as a float
        /// </summary>
        /// <returns></returns>
        public float Accuracy()
        {
            float result = this.truePositives + this.trueNegatives;
            result /= this.truePositives + this.trueNegatives + this.falsePositives + this.falseNegatives;
            return result;
        }

        /// <summary>
        /// Calculates and returns model Precision as a float
        /// </summary>
        /// <returns></returns>
        public float Precision()
        {
            float result = this.truePositives / (this.truePositives + this.falsePositives);
            return result;
        }

        /// <summary>
        /// Calculates and returns model Specificity as a float
        /// </summary>
        /// <returns></returns>
        public float Specificity()
        {
            float result = this.trueNegatives / (this.trueNegatives + this.falseNegatives);
            return result;
        }

        /// <summary>
        /// Calculates and returns model F1_score as a float
        /// </summary>
        /// <returns></returns>
        public float F1_Score()
        {
            float result = 2 * this.truePositives;
            result /= 2 * this.truePositives + this.falsePositives + this.falseNegatives;
            return result;
        }

        /// <summary>
        /// Calculates and prints all the implemented model metrics
        /// Included metrics:
        /// - Model Accuracy
        /// - Model Precision
        /// - Model Specificity
        /// - Model F1_score
        /// - Model Confusion matrix
        /// </summary>
        public void PrintMeasures()
        {
            Console.WriteLine(string.Format("Model accuracy is:\t{0:0.00}", this.Accuracy()));
            Console.WriteLine(string.Format("Model precision is:\t{0:0.00}", this.Precision()));
            Console.WriteLine(string.Format("Model specificity is:\t{0:0.00}", this.Specificity()));
            Console.WriteLine(string.Format("Model F1_score is:\t{0:0.00}", this.F1_Score()));
            Console.WriteLine("Confusion matrix is:");
            Console.WriteLine("[[{0}, {1}],", this.truePositives, this.falsePositives);
            Console.WriteLine(" [{0}, {1}]]", this.falseNegatives, this.trueNegatives);
        }
    }
}
