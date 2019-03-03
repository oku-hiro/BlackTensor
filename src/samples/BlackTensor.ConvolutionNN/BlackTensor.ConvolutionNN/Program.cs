using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;

namespace BlackTensor.ConvolutionNN
{
    class Program
    {
        static void Main(string[] args)
        {
            string image_file = "train-images.idx3-ubyte";
            string label_file = "train-labels.idx1-ubyte";
            string t_image_file = "t10k-images.idx3-ubyte";
            string t_label_file = "t10k-labels.idx1-ubyte";

            FileStream fs1 = new FileStream(image_file, FileMode.Open);
            BinaryReader byte_image = new BinaryReader(fs1);

            FileStream fs2 = new FileStream(label_file, FileMode.Open);
            BinaryReader byte_label = new BinaryReader(fs2);

            byte[][] i_data = new byte[60000][];
            byte[][] image_data = new byte[60000][];
            byte[] l_data = new byte[60000];
            byte[] label_data = new byte[60000];

            for (int i = 0; i < 60000; i++)
            {
                i_data[i] = new byte[784];
                image_data[i] = new byte[784];
            }

            for (int j = 0; j < 60000; j++)
            {
                l_data[j] = byte_label.ReadByte();
                for (int i = 0; i < 784; i++)
                {
                    i_data[j][i] = byte_image.ReadByte();
                }
            }

            for (int k = 0; k < 60000 - 8; k++)
            {
                label_data[k] = l_data[k + 8];

                for (int j = 0; j < 28; j++)
                {
                    for (int i = 0; i < 14; i++)
                    {
                        int p = i + j * 28;
                        int q = (i + 14) + j * 28;
                        image_data[k][q] = i_data[k][p];
                    }

                    for (int i = 14; i < 28; i++)
                    {
                        int p = i + j * 28;
                        int q = (i - 14) + j * 28;
                        image_data[k][q] = i_data[k][p];
                    }
                }
            }

            LearningParameter lp;
            lp.batch_sample = 10;
            lp.dense_unit = 0;
            lp.epochs = 10000;
            lp.data_sample = 5000;
            lp.input_channel = 1;
            lp.input_x = 28;
            lp.input_y = 28;
            lp.optimizer = 2;
            lp.lr = Math.Pow(10.0, -4.0);

            lp.output_data = new double[5000][];
            lp.input_data = new double[5000][];
            for (int i = 0; i < 5000; i++)
            {
                lp.output_data[i] = new double[10];
                lp.input_data[i] = new double[784];
            }

            for (int j = 0; j < 5000; j++)
            {
                lp.output_data[j][label_data[j]] = 1.0;
                for (int i = 0; i < 784; i++)
                {
                    lp.input_data[j][i] = image_data[j][i];
                    lp.input_data[j][i] /= 255.0;
                }
            }

            BlackTensor alg = new BlackTensor();

            alg.Conv2d(1, 4, 5, 5);
            alg.Activation(1);

            alg.Conv2d(1, 6, 5, 5);
            alg.Activation(1);

            alg.Pooling(2, 2);

            alg.Normalization(5, 5, 0);

            alg.Conv2d(1, 8, 5, 5);
            alg.Activation(1);

            alg.Pooling(2, 2);

            alg.Dense(50, 0, 0, 0);
            alg.Activation(1);

            alg.Dense(50, 0, 0, 0);
            alg.Activation(1);

            alg.Dense(10, 0, 0, 0);
            alg.Activation(2);

            alg.Learning(lp);

            FileStream fs3 = new FileStream(t_image_file, FileMode.Open);
            BinaryReader t_byte_image = new BinaryReader(fs3);

            FileStream fs4 = new FileStream(t_label_file, FileMode.Open);
            BinaryReader t_byte_label = new BinaryReader(fs4);

            for (int j = 0; j < 10000; j++)
            {
                l_data[j] = t_byte_label.ReadByte();
                for (int i = 0; i < 784; i++)
                {
                    i_data[j][i] = t_byte_image.ReadByte();
                }
            }

            for (int k = 0; k < 10000 - 8; k++)
            {
                label_data[k] = l_data[k + 8];

                for (int j = 0; j < 28; j++)
                {
                    for (int i = 0; i < 14; i++)
                    {
                        int p = i + j * 28;
                        int q = (i + 14) + j * 28;
                        image_data[k][q] = i_data[k][p];
                    }

                    for (int i = 14; i < 28; i++)
                    {
                        int p = i + j * 28;
                        int q = (i - 14) + j * 28;
                        image_data[k][q] = i_data[k][p];
                    }
                }
            }

            double[] test = new double[784];
            for (int i = 0; i < 784; i++)
            {
                test[i] = image_data[0][i];
                test[i] /= 255.0;
            }

            double[] result = alg.Evaluate(test);

            for (int i = 0; i < result.Length; i++)
            {
                Console.WriteLine(result[i]);
            }
            Console.ReadLine();
        }
    }
}
