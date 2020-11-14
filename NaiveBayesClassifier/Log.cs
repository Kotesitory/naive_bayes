using System;
using System.Collections.Generic;
using System.Text;

namespace NaiveBayesClassifier
{
    /// <summary>
    /// Helper class for loging
    /// </summary>
    class Log
    {
        public static void Error(string from, string message)
        {
            Console.WriteLine("======= ERROR =======");
            Console.WriteLine(string.Format("{0}: {1}", from, message));
        }

        public static void Warning(string from, string message)
        {
            Console.WriteLine("======= WARNING =======");
            Console.WriteLine(string.Format("{0}: {1}", from, message));
        }
    }
}
