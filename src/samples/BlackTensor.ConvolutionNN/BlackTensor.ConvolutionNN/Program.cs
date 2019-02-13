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
            string image_file = ".\\train-images.idx3-ubyte";
            string label_file = ".\\train-labels.idx1-ubyte";

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

            double[][][] image_data = new double[10][][];
            for (int k = 0; k < 10; k++)
            {
                image_data[k] = new double[500][];
                for (int j = 0; j < 500; j++)
                {
                    image_data[k][j] = new double[784];
                }
            }

            int[] step = new int[10];
            for (int k = 0; k < 6000; k++)
            {
                int s = label_data2[k];
                if (step[s] < 500)
                {
                    for (int i = 0; i < 784; i++)
                    {
                        image_data[s][step[s]][i] = byte_data2[k][i];
                    }
                }
                step[s] += 1;
            }

            LearningParameter lp;
            lp.batch_sample = 10;
            lp.dense_unit = 0;
            lp.epochs = 100000;
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

            int m = 0;
            for (int k = 0; k < 500; k++)
            {
                for (int j = 0; j < 10; j++)
                {
                    lp.output_data[m][j] = 1.0;
                    for (int i = 0; i < 784; i++)
                    {
                        lp.input_data[m][i] = image_data[j][k][i];
                        lp.input_data[m][i] /= 255.0;
                    }
                    m += 1;
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
        }
    }
}
