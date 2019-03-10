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
            var imageFile = "train-images.idx3-ubyte";
            var labelFile = "train-labels.idx1-ubyte";
            var tImageFile = "t10k-images.idx3-ubyte";
            var tLabelFile = "t10k-labels.idx1-ubyte";

            var iData = new byte[60000][];
            var imageData = new byte[60000][];
            var lData = new byte[60000];
            var labelData = new byte[60000];

            for (var i = 0; i < 60000; i++)
            {
                iData[i] = new byte[784];
                imageData[i] = new byte[784];
            }

            using (var fs1 = new FileStream(imageFile, FileMode.Open))
            using (var fs2 = new FileStream(labelFile, FileMode.Open))
            {
                var byteImage = new BinaryReader(fs1);
                var byteLabel = new BinaryReader(fs2);

                for (var j = 0; j < 60000; j++)
                {
                    lData[j] = byteLabel.ReadByte();
                    for (var i = 0; i < 784; i++)
                    {
                        iData[j][i] = byteImage.ReadByte();
                    }
                }
            }

            for (var k = 0; k < 60000 - 8; k++)
            {
                labelData[k] = lData[k + 8];

                for (var j = 0; j < 28; j++)
                {
                    for (var i = 0; i < 14; i++)
                    {
                        var p = i + j * 28;
                        var q = (i + 14) + j * 28;
                        imageData[k][q] = iData[k][p];
                    }

                    for (var i = 14; i < 28; i++)
                    {
                        var p = i + j * 28;
                        var q = (i - 14) + j * 28;
                        imageData[k][q] = iData[k][p];
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
            for (var i = 0; i < 5000; i++)
            {
                lp.output_data[i] = new double[10];
                lp.input_data[i] = new double[784];
            }

            for (var j = 0; j < 5000; j++)
            {
                lp.output_data[j][labelData[j]] = 1.0;
                for (var i = 0; i < 784; i++)
                {
                    lp.input_data[j][i] = imageData[j][i];
                    lp.input_data[j][i] /= 255.0;
                }
            }

            var alg = new BlackTensor();

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

            using (var fs3 = new FileStream(tImageFile, FileMode.Open))
            using (var fs4 = new FileStream(tLabelFile, FileMode.Open))
            {
                var tByteImage = new BinaryReader(fs3);
                var tByteLabel = new BinaryReader(fs4);

                for (var j = 0; j < 10000; j++)
                {
                    lData[j] = tByteLabel.ReadByte();
                    for (var i = 0; i < 784; i++)
                    {
                        iData[j][i] = tByteImage.ReadByte();
                    }
                }
            }

            for (var k = 0; k < 10000 - 8; k++)
            {
                labelData[k] = lData[k + 8];

                for (var j = 0; j < 28; j++)
                {
                    for (var i = 0; i < 14; i++)
                    {
                        var p = i + j * 28;
                        var q = (i + 14) + j * 28;
                        imageData[k][q] = iData[k][p];
                    }

                    for (var i = 14; i < 28; i++)
                    {
                        var p = i + j * 28;
                        var q = (i - 14) + j * 28;
                        imageData[k][q] = iData[k][p];
                    }
                }
            }

            var test = new double[784];
            for (var i = 0; i < 784; i++)
            {
                test[i] = imageData[0][i];
                test[i] /= 255.0;
            }

            var result = alg.Evaluate(test);

            foreach (var t in result)
            {
                Console.WriteLine(t);
            }
            Console.ReadLine();
        }
    }
}
