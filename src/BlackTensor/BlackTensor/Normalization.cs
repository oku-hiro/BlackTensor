using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace BlackTensor
{
    class Normalization
    {
        public int input_unit;
        public int input_channel;
        public int output_unit;
        public int output_channel;
        public int input_x;
        public int input_y;
        public int size_x;
        public int size_y;
        public int batch_sample;

        public double[][] input;
        public double[][] output;
        public double[][] next_delta;
        public double[][] this_delta;
        public double[][] pre_grad;
        public double[][] this_grad;

        double eps = Math.Pow(10.0, -8.0);
        int input_xy;
        int size_xy;

        public Normalization()
        {}

        public void Setting()
        {
            input_xy = input_x * input_y;
            size_xy = size_x * size_y;
            output_channel = input_channel;
            output_unit = input_unit;

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

        public void Batch1D()
        {
            for (int i = 0; i < input_unit; i++)
            {
                double average = 0.0;
                for (int b = 0; b < batch_sample; b++)
                {
                    average += input[b][i];
                }
                average /= batch_sample;

                double sigma1 = 0.0;
                double sigma2 = 0.0;
                for (int b = 0; b < batch_sample; b++)
                {
                    sigma1 = input[b][i] - average;
                    sigma2 += sigma1 * sigma1;
                }
                sigma2 /= batch_sample;
                sigma1 = Math.Sqrt(sigma2 + eps);

                for (int b = 0; b < batch_sample; b++)
                {
                    output[b][i] = (input[b][i] - average) / sigma1;
                    this_grad[b][i] = 1.0;
                }
            }
        }

        public void Subtractive()
        {
            for (int b = 0; b < batch_sample; b++)
            {
                for (int k = 0; k < input_channel; k++)
                {
                    int ki = k * input_xy;
                    for (int j = 0; j < input_y; j++)
                    {
                        for (int i = 0; i < input_x; i++)
                        {
                            double pixel = 0;
                            double average = 0;
                            for (int y = -size_y / 2; y <= size_y / 2; y++)
                            {
                                for (int x = -size_x / 2; x <= size_x / 2; x++)
                                {
                                    if (0 <= j + y && j + y < input_y)
                                    {
                                        if (0 <= i + x && i + x < input_x)
                                        {
                                            average += input[b][(i + x) + (j + y) * input_x + ki];
                                            pixel += 1;
                                        }
                                    }
                                }
                            }
                            average /= pixel;
                            int p = i + j * input_x + ki;
                            output[b][p] = input[b][p] - average;
                            this_grad[b][p] = 1.0;
                        }
                    }
                }
            }
        }

        public void Divisive()
        {
            for (int b = 0; b < batch_sample; b++)
            {
                for (int k = 0; k < input_channel; k++)
                {
                    int ki = k * input_xy;
                    for (int j = 0; j < input_y; j++)
                    {
                        for (int i = 0; i < input_x; i++)
                        {
                            double pixel = 0.0;
                            double average = 0.0;
                            for (int y = -size_y / 2; y <= size_y / 2; y++)
                            {
                                for (int x = -size_x / 2; x <= size_x / 2; x++)
                                {
                                    if (0 <= j + y && j + y < input_y)
                                    {
                                        if (0 <= i + x && i + x < input_x)
                                        {
                                            average += input[b][(i + x) + (j + y) * input_x + ki];
                                            pixel += 1;
                                        }
                                    }
                                }
                            }
                            average /= pixel;

                            double sigma1 = 0.0;
                            double sigma2 = 0.0;
                            for (int y = -size_y / 2; y <= size_y / 2; y++)
                            {
                                for (int x = -size_x / 2; x <= size_x / 2; x++)
                                {
                                    if (0 <= j + y && j + y < input_y)
                                    {
                                        if (0 <= i + x && i + x < input_x)
                                        {
                                            sigma1 = input[b][(i + x) + (j + y) * input_x + ki] - average;
                                            sigma2 += sigma1 * sigma1;
                                        }
                                    }
                                }
                            }

                            sigma2 /= pixel;
                            sigma1 = Math.Sqrt(sigma2);
                            if (sigma1 < 1.0)
                            {
                                sigma1 = 1.0;
                            }

                            int p = i + j * input_x + ki;
                            output[b][p] = (input[b][p] - average) / sigma1;
                            this_grad[b][p] = 1.0;
                        }
                    }
                }
            }
        }

        public void Delta_Propagation()
        {
            Parallel.For(0, batch_sample, b =>
            {
                for (int i = 0; i < input_unit; i++)
                {
                    next_delta[b][i] = this_delta[b][i] * pre_grad[b][i];
                }
            });
        }
    }
}