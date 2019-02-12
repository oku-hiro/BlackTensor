using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;

namespace BlackTensor
{
    class Dense
    {
        public int input_unit;
        public int output_unit;
        public int batch_sample;
        public double lr;

        public double[][] input;
        public double[][] output;
        public double[][] next_delta;
        public double[][] this_delta;
        public double[][] pre_grad;
        public double[][] this_grad;

        double[][] w;
        double[][] dw;
        double[][] m;
        double[][] v;
        double[] u;
       
        const double beta1 = 0.9;
        const double beta2 = 0.999;
        const double gamma = 0.99;
        double epsilon = Math.Pow(10.0, -8.0);
        double b1, b2;

        public Dense()
        {
            b1 = beta1;
            b2 = beta2;
        }

        public void Setting()
        {
            Random rnd = new Random();
            input = new double[batch_sample][];
            output = new double[batch_sample][];
            next_delta = new double[batch_sample][];
            this_delta = new double[batch_sample][];
            pre_grad = new double[batch_sample][];
            this_grad = new double[batch_sample][];

            for (int i = 0; i < batch_sample; i++)
            {
                input[i] = new double[input_unit + 1];
                input[i][0] = 1.0;
                output[i] = new double[output_unit];
                next_delta[i] = new double[input_unit + 1];
                this_delta[i] = new double[output_unit];
                pre_grad[i] = new double[input_unit + 1];
                this_grad[i] = new double[output_unit];
            }

            u = new double[output_unit];
            m = new double[output_unit][];
            v = new double[output_unit][];
            for (int i = 0; i < output_unit; i++)
            {
                m[i] = new double[input_unit + 1];
                v[i] = new double[input_unit + 1];
            }

            for (int j = 0; j < output_unit; j++)
            {
                for (int i = 0; i <= input_unit; i++)
                {
                    m[j][i] = 0.0;
                    v[j][i] = 0.0;
                }
            }

            w = new double[output_unit][];
            dw = new double[output_unit][];

            for (int i = 0; i < output_unit; i++)
            {
                w[i] = new double[input_unit + 1];
                dw[i] = new double[input_unit + 1];
            }

            for (int j = 0; j < output_unit; j++)
            {
                for (int i = 0; i <= input_unit; i++)
                {
                    w[j][i] = rnd.NextDouble();
                    w[j][i] *= Math.Pow(10.0, -5.0); 
                }
            }
        }

        public void Process()
        {
            for (int b = 0; b < batch_sample; b++)
            {
                for (int j = 0; j < output_unit; j++)
                {
                    output[b][j] = 0.0;
                    for (int i = 0; i <= input_unit; i++)
                    {
                        output[b][j] += w[j][i] * input[b][i];
                    }
                    this_grad[b][j] = 1.0;
                }
            }
        }

        public void Delta_Propagation()
        {
            for (int b = 0; b < batch_sample; b++)
            {
                for (int j = 1; j <= input_unit; j++)
                {
                    next_delta[b][j] = 0.0;
                    for (int i = 0; i < output_unit; i++)
                    {
                        next_delta[b][j] += w[i][j] * this_delta[b][i] * pre_grad[b][j];
                    }
                }
            }
        }

        public void Back_Propagation()
        {
            b1 *= beta1;
            b2 *= beta2;
            for (int j = 0; j < output_unit; j++)
            {
                for (int i = 0; i <= input_unit; i++)
                {
                    dw[j][i] = 0.0;
                }
            }

            for (int b = 0; b < batch_sample; b++)
            {
                for (int j = 0; j < output_unit; j++)
                {
                    for (int i = 0; i <= input_unit; i++)
                    {
                        dw[j][i] += this_delta[b][j] * input[b][i];
                    }
                }
            }
        }

        public void ADAM()
        {
            for (int j = 0; j < output_unit; j++)
            {
                for (int i = 0; i <= input_unit; i++)
                {
                    m[j][i] = beta1 * m[j][i] + (1.0 - beta1) * dw[j][i];
                    v[j][i] = beta2 * v[j][i] + (1.0 - beta2) * dw[j][i] * dw[j][i];
                    double a = m[j][i] / (1.0 - b1);
                    double b = v[j][i] / (1.0 - b2);
                    w[j][i] -= lr * a / (Math.Sqrt(b) + epsilon);
                }
            }
        }

        public void RmsProp()
        {
            for (int j = 0; j < output_unit; j++)
            {
                for (int i = 0; i <= input_unit; i++)
                {
                    m[j][i] = beta1 * m[j][i] + (1.0 - gamma) * dw[j][i] * dw[j][i];
                    w[j][i] -= lr * dw[j][i] / (Math.Sqrt(m[j][i]) + epsilon);
                }
            }
        }

        public void SGD()
        {
            for (int j = 0; j < output_unit; j++)
            {
                for (int i = 0; i <= input_unit; i++)
                {
                    w[j][i] -= lr * dw[j][i];
                }
            }
        }

        public void SaveParameter(int layer)
        {
            StreamWriter sw = new StreamWriter("fc" + (layer + 1).ToString());
            for (int j = 0; j < output_unit; j++)
            {
                for (int i = 0; i <= input_unit; i++)
                {
                    sw.WriteLine(w[j][i]);
                }
            }
            sw.Close();
        }
    }
}
