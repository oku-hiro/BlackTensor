using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace BlackTensor
{
    class GaussianDistribution
    {
        public int input_unit;
        public int output_unit;
        public int batch_sample;

        public double[][] input;
        public double[][] output;
        public double[][] next_delta;
        public double[][] this_delta;
        public double[][] pre_grad;
        public double[][] this_grad;

        Random rnd = new Random();
        double pi = Math.PI;

        public GaussianDistribution()
        {}

        public void Setting()
        {
            input = new double[batch_sample][];
            output = new double[batch_sample][];
            next_delta = new double[batch_sample][];
            this_delta = new double[batch_sample][];
            pre_grad = new double[batch_sample][];
            this_grad = new double[batch_sample][];

            for (int i = 0; i < batch_sample; i++)
            {
                input[i] = new double[input_unit];
                output[i] = new double[output_unit];
                next_delta[i] = new double[input_unit];
                this_delta[i] = new double[output_unit];
                pre_grad[i] = new double[input_unit];
                this_grad[i] = new double[output_unit];
            }
        }

        public void Process()
        {
            for (int j = 0; j < batch_sample; j++)
            {
                double z = Math.Sqrt(-2.0 * Math.Log(rnd.NextDouble())) * Math.Cos(2.0 * pi * rnd.NextDouble());
                for (int i = 0; i < input_unit / 2; i++)
                {
                    output[j][i] = input[j][2 * i] + z * input[j][2 * i + 1];
                    this_grad[j][i] = 1.0;
                }
            }
        }

        public void Delta_Propagation()
        {
            for (int b = 0; b < batch_sample; b++)
            {
                for (int j = 0; j < input_unit / 2; j++)
                {
                    for (int i = 2 * j; i < 2 * (j + 1); i++)
                    {
                        next_delta[b][i] = this_delta[b][j] * pre_grad[b][i];
                    }
                }
            }
        }
    }
}