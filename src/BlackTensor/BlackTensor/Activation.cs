using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace BlackTensor
{
    class Activation
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

        public void Sigmoid()
        {
            for (int b = 0; b < batch_sample; b++)
            {
                for (int i = 0; i < output_unit; i++)
                {
                    output[b][i] = 1.0 / (1.0 + Math.Exp(-input[b][i]));
                    this_grad[b][i] = output[b][i] * (1.0 - output[b][i]);
                }
            }
        }

        public void ReLU()
        {
            for (int b = 0; b < batch_sample; b++)
            {
                for (int i = 0; i < output_unit; i++)
                {
                    if (input[b][i] < 0.0)
                    {
                        output[b][i] = 0.0;
                        this_grad[b][i] = 0.0;
                    }
                    else
                    {
                        output[b][i] = input[b][i];
                        this_grad[b][i] = 1.0;
                    }
                }
            }
        }

        public void Tanh()
        {
            for (int b = 0; b < batch_sample; b++)
            {
                for (int i = 0; i < output_unit; i++)
                {
                    output[b][i] = Math.Tanh(input[b][i]);
                    this_grad[b][i] = 1.0 - output[b][i] * output[b][i];
                }
            }
        }

        public void Softmax()
        {
            for (int b = 0; b < batch_sample; b++)
            {
                for (int i = 0; i < output_unit; i++)
                {
                    output[b][i] = Math.Exp(input[b][i]);
                }

                double sum = 0.0;
                for (int i = 0; i < output_unit; i++)
                {
                    sum += output[b][i];
                }

                for (int i = 0; i < output_unit; i++)
                {
                    output[b][i] /= sum;
                    this_grad[b][i] = 1.0;
                }
            }
        }

        public void Delta_Propagation()
        {
            for (int b = 0; b < batch_sample; b++)
            {
                for (int i = 0; i < input_unit; i++)
                {
                    next_delta[b][i] = this_delta[b][i] * pre_grad[b][i];
                }
            }
        }
    }
}
