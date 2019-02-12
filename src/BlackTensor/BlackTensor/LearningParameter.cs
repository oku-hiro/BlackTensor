using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace BlackTensor
{
    public struct LearningParameter
    {
        public int epochs;
        public int input_channel;
        public int input_x;
        public int input_y;
        public int dense_unit;
        public int batch_sample;
        public int data_sample;
        public int optimizer;
        public double lr;
        public double[][] input_data;
        public double[][] output_data;
    }
}
