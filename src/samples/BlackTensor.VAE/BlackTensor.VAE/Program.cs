using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;
using System.Drawing;

namespace BlackTensor.VAE
{
    class Program
    {
        static void Main(string[] args)
        {
            string image_file = ".train-images.idx3-ubyte";
            string label_file = ".train-labels.idx1-ubyte";

            FileStream fs1 = new FileStream(image_file, FileMode.Open);
            BinaryReader byte_image = new BinaryReader(fs1);

            FileStream fs2 = new FileStream(label_file, FileMode.Open);
            BinaryReader byte_label = new BinaryReader(fs2);

            byte[][] byte_data1 = new byte[60000][];
            byte[][] byte_data2 = new byte[60000][];
            byte[] label_data1 = new byte[60000];
            byte[] label_data2 = new byte[60000];

            for (int i = 0; i < 60000; i++)
            {
                byte_data1[i] = new byte[784];
                byte_data2[i] = new byte[784];
            }

            for (int j = 0; j < 60000; j++)
            {
                label_data1[j] = byte_label.ReadByte();
                for (int i = 0; i < 784; i++)
                {
                    byte_data1[j][i] = byte_image.ReadByte();
                }
            }

            for (int k = 0; k < 60000 - 8; k++)
            {
                label_data2[k] = label_data1[k + 8];

                for (int j = 0; j < 28; j++)
                {
                    for (int i = 0; i < 14; i++)
                    {
                        int p = i + j * 28;
                        int q = (i + 14) + j * 28;
                        byte_data2[k][q] = byte_data1[k][p];
                    }

                    for (int i = 14; i < 28; i++)
                    {
                        int p = i + j * 28;
                        int q = (i - 14) + j * 28;
                        byte_data2[k][q] = byte_data1[k][p];
                    }
                }
            }

            double[][] image_data = new double[5000][];
            for (int i = 0; i < 5000; i++)
            {
                image_data[i] = new double[784];
            }

            for (int j = 0; j < 6000; j++)
            {
                for (int i = 0; i < 784; i++)
                {
                    image_data[j][i] = byte_data2[j][i];
                }
            }

            LearningParameter lp;
            lp.batch_sample = 16;
            lp.dense_unit = 0;
            lp.epochs = 100000;
            lp.data_sample = 500;
            lp.input_channel = 1;
            lp.input_x = 28;
            lp.input_y = 28;
            lp.optimizer = 0;
            lp.lr = Math.Pow(10.0, -2.0);

            lp.output_data = new double[5000][];
            lp.input_data = new double[5000][];
            for (int i = 0; i < 5000; i++)
            {
                lp.output_data[i] = new double[784];
                lp.input_data[i] = new double[784];
            }

            for (int j = 0; j < 5000; j++)
            {
                for (int i = 0; i < 784; i++)
                {
                    lp.input_data[j][i] = image_data[j][i];
                    lp.input_data[j][i] /= 255.0;
                    lp.output_data[j][i] = image_data[j][i];
                    lp.output_data[j][i] /= 255.0;
                }
            }

            BlackTensor alg = new BlackTensor();

            alg.Dense(392, 0, 0, 0);
            alg.Activation(0);

            alg.Dense(32, 0, 0, 0);
            alg.Activation(0);

            alg.GaussianDistribution();

            alg.Dense(32, 0, 0, 0);
            alg.Activation(0);

            alg.Dense(784, 0, 0, 0);
            alg.Activation(0);

            alg.Learning(lp);

            double[] test = new double[784];
            for (int i = 0; i < 784; i++)
            {
                test[i] = image_data[0][i];
                test[i] = 255.0;
            }

            double[] output = alg.Evaluate(test);
            Bitmap map = new Bitmap(28, 28);
            for (int j = 0; j < 28; j++)
            {
                for (int i = 0; i < 28; i++)
                {
                    int d = (int)(255.0 * output[i] + 0.5);
                    map.SetPixel(i, j, Color.FromArgb(d, d, d));
                }
            }
            map.Save("test.png");
        }
    }
}