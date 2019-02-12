using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace BlackTensor.ConvolutionNN
{
    class Program
    {
        static void Main(string[] args)
        {
            double[][][] image_data = new double[10][][];
            for (int j = 0; j < 10; j++)
            {
                image_data[j] = new double[600][];
                for (int i = 0; i < 600; i++)
                {
                    image_data[j][i] = new double[784];
                }
            }

            //.\\MNIST\\number0\\image100.bmp　
            //データの取得先は、http://yann.lecun.com/exdb/mnist/
            //をbitmapに変えて数字ごとのフォルダに格納しました。

            Bitmap map;
            for (int k = 0; k < 10; k++)
            {
                for (int j = 0; j < 600; j++)
                {
                    string file_name = ".\\MNIST\\number" + k.ToString() + "\\image" + j.ToString() + ".bmp";
                    map = new Bitmap(file_name);

                    int n = 0;
                    for (int y = 0; y < 28; y++)
                    {
                        for (int x = 0; x < 28; x++)
                        {
                            image_data[k][j][n] = map.GetPixel(x, y).R;
                            n += 1;
                        }
                    }
                }
            }

            LearningParameter lp;
            lp.batch_sample = 10;
            lp.dense_unit = 0;
            lp.epochs = 375000;
            lp.data_sample = 6000;
            lp.input_channel = 1;
            lp.input_x = 28;
            lp.input_y = 28;
            lp.optimizer = 2;
            lp.lr = Math.Pow(10.0, -4.0);

            lp.output_data = new double[6000][];
            lp.input_data = new double[6000][];
            for (int i = 0; i < 6000; i++)
            {
                lp.output_data[i] = new double[10];
                lp.input_data[i] = new double[784];
            }

            int m = 0;
            for (int k = 0; k < 600; k++)
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
