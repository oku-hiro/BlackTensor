using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;

namespace BlackTensor
{
    class Conv2dTranspose
    {
        public int input_channel;
        public int input_x;
        public int input_y;
        public int input_unit;
        public int output_channel;
        public int output_x;
        public int output_y;
        public int output_unit;
        public int filter_channel;
        public int filter_x;
        public int filter_y;
        public int batch_sample;

        public double lr;
        public double[][] input;
        public double[][] this_delta;
        public double[][] next_delta;
        public double[][] pre_grad;
        public double[][] this_grad;
        public double[][] output;

        int input_xy;
        int output_xy;
        int filter_xy;
        int filter_element;
        int pad_x;
        int pad_y;
        int pad_xy;
        int pad_unit;

        double[][] padding;
        double[][] padding_delta;
        double[][] padding_grad;
        double[] filter;
        double[] d_filter;
        double[] bias;
        double[] d_bias;
        double[] fm;
        double[] fv;
        double[] bm;
        double[] bv;
        int[][] connection;

        const int upsampling = 2;
        const double beta1 = 0.9;
        const double beta2 = 0.999;
        const double gamma = 0.99;
        double b1, b2;
        double epsilon = Math.Pow(10.0, -8.0);

        public Conv2dTranspose()
        {
            b1 = beta1;
            b2 = beta2;
        }

        public void Setting()
        {
            Random rnd = new Random();

            input_xy = input_x * input_y;
            output_channel = filter_channel;
            pad_x = upsampling * input_x;
            pad_y = upsampling * input_y;
            pad_xy = pad_x * pad_y;
            output_x = pad_x;
            output_y = pad_y;
            output_xy = output_x * output_y;

            filter_xy = filter_x * filter_y;
            input_unit = input_channel * input_xy;
            output_unit = output_channel * output_xy;
            filter_element = input_channel * filter_channel * filter_xy;
            pad_unit = input_channel * pad_xy;

            input = new double[batch_sample][];
            padding = new double[batch_sample][];
            output = new double[batch_sample][];
            next_delta = new double[batch_sample][];
            this_delta = new double[batch_sample][];
            padding_delta = new double[batch_sample][];
            padding_grad = new double[batch_sample][];
            pre_grad = new double[batch_sample][];
            this_grad = new double[batch_sample][];

            for (int i = 0; i < batch_sample; i++)
            {
                input[i] = new double[input_unit];
                padding[i] = new double[pad_unit];
                output[i] = new double[output_unit];
                next_delta[i] = new double[input_unit];
                this_delta[i] = new double[output_unit];
                padding_delta[i] = new double[pad_unit];
                padding_grad[i] = new double[pad_unit];
                pre_grad[i] = new double[input_unit];
                this_grad[i] = new double[output_unit];
            }

            filter = new double[filter_element];
            d_filter = new double[filter_element];
            bias = new double[filter_channel];
            d_bias = new double[filter_channel];

            connection = new int[output_unit][];
            for (int i = 0; i < output_unit; i++)
            {
                connection[i] = new int[pad_unit];
            }

            fm = new double[filter_element];
            fv = new double[filter_element];
            bm = new double[filter_channel];
            bv = new double[filter_channel];

            for (int i = 0; i < filter_element; i++)
            {
                fm[i] = 0.0;
                fv[i] = 0.0;
            }

            for (int i = 0; i < filter_channel; i++)
            {
                bm[i] = 0.0;
                bv[i] = 0.0;
            }

            for (int i = 0; i < filter_element; i++)
            {
                filter[i] = rnd.NextDouble();
                filter[i] *= Math.Pow(10.0, -8.0);
            }

            for (int i = 0; i < filter_channel; i++)
            {
                bias[i] = 0.0;
            }

            for (int j = 0; j < output_unit; j++)
            {
                for (int i = 0; i < pad_unit; i++)
                {
                    connection[j][i] = -1;
                }
            }

            for (int fk = 0; fk < filter_channel; fk++)
            {
                for (int ik = 0; ik < input_channel; ik++)
                {
                    for (int iy = 0; iy < pad_y; iy++)
                    {
                        for (int ix = 0; ix < pad_x; ix++)
                        {
                            int f = ik * filter_xy + fk * input_channel * filter_xy;
                            for (int fy = -filter_y / 2; fy <= filter_y / 2; fy++)
                            {
                                for (int fx = -filter_x / 2; fx <= filter_x / 2; fx++)
                                {
                                    if (0 <= ix + fx && ix + fx < pad_x)
                                    {
                                        if (0 <= iy + fy && iy + fy < pad_y)
                                        {
                                            int p = ix + iy * output_x + fk * output_xy;         //output unit
                                            int q = (ix + fx) + (iy + fy) * pad_x + ik * pad_xy; //input unit
                                            connection[p][q] = f;
                                        }
                                    }
                                    f += 1;
                                }
                            }
                        }
                    }
                }
            }
        }

        public void Process()
        {
            Parallel.For(0, batch_sample, b =>
            {
                for (int k = 0; k < input_channel; k++)
                {
                    for (int j = 0; j < input_y; j++)
                    {
                        for (int i = 0; i < input_x; i++)
                        {
                            padding[b][upsampling * (i + j * pad_x) + k * pad_xy] = input[b][i + j * input_x + k * input_xy];
                        }
                    }
                }

                for (int j = 0; j < output_unit; j++)
                {
                    output[b][j] = 0.0;
                    for (int i = 0; i < pad_unit; i++)
                    {
                        if (connection[j][i] > -1)
                        {
                            output[b][j] += filter[connection[j][i]] * padding[b][i];
                        }
                    }
                    output[b][j] += bias[j / output_xy];
                    this_grad[b][j] = 1.0;
                }
            });
        }

        public void DeltaPropagation()
        {
            Parallel.For(0, batch_sample, b =>
            {
                for (int k = 0; k < input_channel; k++)
                {
                    for (int j = 0; j < input_y; j++)
                    {
                        for (int i = 0; i < input_x; i++)
                        {
                            padding_grad[b][upsampling * (i + j * pad_x) + k * pad_xy] = pre_grad[b][i + j * input_x + k * input_xy];
                        }
                    }
                }

                for (int j = 0; j < pad_unit; j++)
                {
                    padding_delta[b][j] = 0.0;
                    for (int i = 0; i < output_unit; i++)
                    {
                        if (connection[i][j] > -1)
                        {
                            padding_delta[b][j] += filter[connection[i][j]] * this_delta[b][i] * padding_grad[b][j];
                        }
                    }
                }

                for (int k = 0; k < input_channel; k++)
                {
                    for (int j = 0; j < input_y; j++)
                    {
                        for (int i = 0; i < input_x; i++)
                        {
                            next_delta[b][i + j * input_x + k * input_xy] = padding_delta[b][upsampling * (i + j * pad_x) + k * pad_xy];
                        }
                    }
                }
            });
        }

        public void BackPropagation()
        {
            b1 *= beta1;
            b2 *= beta2;

            for (int i = 0; i < filter_element; i++)
            {
                d_filter[i] = 0.0;
            }

            for (int i = 0; i < filter_channel; i++)
            {
                d_bias[i] = 0.0;
            }

            for (int k = 0; k < batch_sample; k++)
            {
                for (int j = 0; j < output_unit; j++)
                {
                    for (int i = 0; i < pad_unit; i++)
                    {
                        if (connection[j][i] > -1)
                        {
                            d_filter[connection[j][i]] += this_delta[k][j] * padding[k][i];
                        }
                    }
                }

                for (int j = 0; j < filter_channel; j++)
                {
                    for (int i = 0; i < output_xy; i++)
                    {
                        d_bias[j] += this_delta[k][i + j * output_xy];
                    }
                }
            }
        }

        public void SGD()
        {
            for (int i = 0; i < filter_element; i++)
            {
                filter[i] -= lr * d_filter[i];
            }

            for (int i = 0; i < filter_channel; i++)
            {
                bias[i] -= lr * d_bias[i];
            }
        }

        public void ADAM()
        {
            for (int i = 0; i < filter_element; i++)
            {
                fm[i] = beta1 * fm[i] + (1.0 - beta1) * d_filter[i];
                fv[i] = beta2 * fv[i] + (1.0 - beta2) * d_filter[i] * d_filter[i];
                double m = fm[i] / (1.0 - b1);
                double v = fv[i] / (1.0 - b2);
                filter[i] -= lr * m / (Math.Sqrt(v) + epsilon);
            }

            for (int i = 0; i < filter_channel; i++)
            {
                bm[i] = beta1 * bm[i] + (1.0 - beta1) * d_bias[i];
                bv[i] = beta2 * bv[i] + (1.0 - beta2) * d_bias[i] * d_bias[i];
                double m = bm[i] / (1.0 - b1);
                double v = bv[i] / (1.0 - b2);
                bias[i] -= lr * m / (Math.Sqrt(v) + epsilon);
            }
        }

        public void RmsProp()
        {
            for (int i = 0; i < filter_element; i++)
            {
                fv[i] = beta1 * fv[i] + (1.0 - gamma) * d_filter[i] * d_filter[i];
                filter[i] -= lr * d_filter[i] / (Math.Sqrt(fv[i]) + epsilon);
            }

            for (int i = 0; i < filter_channel; i++)
            {
                bv[i] = beta2 * bv[i] + (1.0 - beta2) * d_bias[i] * d_bias[i];
                bias[i] -= lr * d_bias[i] / (Math.Sqrt(bv[i]) + epsilon);
            }
        }

        public void SaveParameter(int layer)
        {
            StreamWriter sw1 = new StreamWriter("conv2d_transpose_filter" + (layer + 1).ToString());
            for (int i = 0; i < filter_element; i++)
            {
                sw1.WriteLine(filter[i]);
            }
            sw1.Close();

            StreamWriter sw2 = new StreamWriter("conv2d_transpose_bias" + (layer + 1).ToString());
            for (int i = 0; i < filter_channel; i++)
            {
                sw2.WriteLine(bias[i]);
            }
            sw2.Close();
        }
    }
}