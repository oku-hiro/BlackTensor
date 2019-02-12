using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace BlackTensor
{
    class Pooling
    {
        public int input_unit;
        public int output_unit;
        public int input_x;
        public int input_y;
        public int size_x;
        public int size_y;
        public int input_channel;
        public int output_channel;
        public int output_x;
        public int output_y;
        public int batch_sample;
        public double[][] input;
        public double[][] output;
        public double[][] next_delta;
        public double[][] this_delta;
        public double[][] pre_grad;
        public double[][] this_grad;

        int input_xy;
        int output_xy;
        int[][] w;

        public Pooling()
        {}

        public void Setting()
        {
            input_xy = input_x * input_y;
            output_channel = input_channel;
            output_x = input_x / size_x;
            output_y = input_y / size_y;
            output_xy = output_x * output_y;
            input_unit = input_channel * input_xy;
            output_unit = output_channel * output_xy;

            input = new double[batch_sample][];
            output = new double[batch_sample][];
            next_delta = new double[batch_sample][];
            this_delta = new double[batch_sample][];
            pre_grad = new double[batch_sample][];
            this_grad = new double[batch_sample][];
            w = new int[batch_sample][];

            for (int i = 0; i < batch_sample; i++)
            {
                input[i] = new double[input_unit];
                output[i] = new double[output_unit];
                next_delta[i] = new double[input_unit];
                this_delta[i] = new double[output_unit];
                pre_grad[i] = new double[input_unit];
                this_grad[i] = new double[output_unit];
                w[i] = new int[output_unit];
            }

        }

        public void Process()
        {
            for (int b = 0; b < batch_sample; b++)
            {
                for (int i = 0; i < output_unit; i++)
                {
                    this_grad[b][i] = 0.0;
                    w[b][i] = -1;
                }
           
                for (int k = 0; k < input_channel; k++)
                {
                    int ki = k * input_xy;
                    int kp = k * output_xy;

                    for (int j = 0; j < output_y; j++)
                    {
                        for (int i = 0; i < output_x; i++)
                        {
                            int start_x = i * size_x;
                            int start_y = j * size_y;
                            int max_x = start_x;
                            int max_y = start_y;

                            double maximum = input[b][0];
                            for (int y = 0; y < size_y; y++)
                            {
                                for (int x = 0; x < size_x; x++)
                                {
                                    if (maximum < input[b][(start_x + x) + (start_y + y) * input_x + ki])
                                    {
                                        max_x = start_x + x;
                                        max_y = start_y + y;
                                        maximum = input[b][max_x + max_y * input_x + ki];
                                    }
                                }
                            }

                            int p = i + j * output_x + kp;
                            w[b][p] = max_x + max_y * input_x + ki;
                            output[b][p] = maximum;
                            this_grad[b][p] = 1.0;
                        }
                    }
                }
            }
        }

        public void Delta_Propagation()
        {
            for (int b = 0; b < batch_sample; b++)
            {
                for (int i = 0; i < input_unit; i++)
                {
                    next_delta[b][i] = 0.0;
                }

                for (int i = 0; i < output_unit; i++)
                {
                    if (w[b][i] > -1)
                    {
                        next_delta[b][w[b][i]] += this_delta[b][i] * pre_grad[b][w[b][i]];
                    }
                }
            }
        }
    }
}