using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace BlackTensor
{
    public class Normalization : BaseAnalysis
    {
        #region 定数
        private readonly double _eps = Math.Pow(10.0, -8.0);
        #endregion

        #region プロパティ
        public Int2D Input2D { get; }
        public Int2D Size2D { get; }

        public int InputChannel { get; }
        public int OutputChannel { get; }
        #endregion

        private readonly int _inputXy;

        #region 初期化
        /// <inheritdoc />
        /// <summary>
        /// 初期化します。
        /// </summary>
        public Normalization() {}

        /// <inheritdoc />
        /// <summary>
        /// 指定した値を使用して、初期化します。
        /// </summary>
        /// <param name="inputOutputUnit"></param>
        /// <param name="batchSample"></param>
        /// <param name="inputX"></param>
        /// <param name="inputY"></param>
        /// <param name="sizeX"></param>
        /// <param name="sizeY"></param>
        /// <param name="inputChannel"></param>
        public Normalization(int inputOutputUnit, int batchSample, int inputX, int inputY, int sizeX, int sizeY, int inputChannel) : base(inputOutputUnit, batchSample)
        {
            this.Input2D = new Int2D(inputX, inputY);
            this.Size2D = new Int2D(sizeX, sizeY);

            this.InputChannel = inputChannel;
            this.OutputChannel = inputChannel;

            this._inputXy = this.Input2D.X * this.Input2D.Y;
        }
        #endregion

        #region メソッド
        public Tuple<double[][], double[][]> Batch1D(double[][] flow, double[][] grad)
        {
            this.SetInputGradData(flow, grad);

            for (var i = 0; i < this.InputUnit; i++)
            {
                var average = 0.0;
                for (var b = 0; b < this.InputOutputData.Input.GetLength(0); b++)
                {
                    average += this.InputOutputData.Input[b][i];
                }
                average /= this.BatchSample;

                var sigma1 = 0.0;
                var sigma2 = 0.0;
                for (var b = 0; b < this.InputOutputData.Input.GetLength(0); b++)
                {
                    sigma1 = this.InputOutputData.Input[b][i] - average;
                    sigma2 += sigma1 * sigma1;
                }
                sigma2 /= this.BatchSample;
                sigma1 = Math.Sqrt(sigma2 + _eps);

                for (var b = 0; b < this.BatchSample; b++)
                {
                    this.InputOutputData.Output[b][i] = (this.InputOutputData.Input[b][i] - average) / sigma1;
                    this.GradData.Output[b][i] = 1.0;
                }
            }

            return new Tuple<double[][], double[][]>(this.InputOutputData.Output, this.GradData.Output);
        }

        public Tuple<double[][], double[][]> Subtractive(double[][] flow, double[][] grad)
        {
            this.SetInputGradData(flow, grad);

            Parallel.For(0, this.InputOutputData.Output.GetLength(0), b =>
            {
                for (var k = 0; k < InputChannel; k++)
                {
                    var ki = k * _inputXy;
                    for (var j = 0; j < this.Input2D.Y; j++)
                    {
                        for (var i = 0; i < this.Input2D.X; i++)
                        {
                            double pixel = 0;
                            double average = 0;
                            for (var y = -this.Size2D.Y / 2; y <= this.Size2D.Y / 2; y++)
                            {
                                for (var x = -this.Size2D.X / 2; x <= this.Size2D.X / 2; x++)
                                {
                                    if (0 > j + y || j + y >= this.Input2D.Y) continue;
                                    if (0 > i + x || i + x >= this.Input2D.X) continue;

                                    average += this.InputOutputData.Input[b][(i + x) + (j + y) * this.Input2D.X + ki];
                                    pixel++;
                                }
                            }
                            average /= pixel;
                            var p = i + j * this.Input2D.X + ki;
                            this.InputOutputData.Output[b][p] = this.InputOutputData.Input[b][p] - average;
                            this.GradData.Output[b][p] = 1.0;
                        }
                    }
                }
            });

            return new Tuple<double[][], double[][]>(this.InputOutputData.Output, this.GradData.Output);
        }

        public Tuple<double[][], double[][]> Divisive(double[][] flow, double[][] grad)
        {
            this.SetInputGradData(flow, grad);

            Parallel.For(0, this.InputOutputData.Output.GetLength(0), b =>
            {
                for (var k = 0; k < InputChannel; k++)
                {
                    var ki = k * _inputXy;
                    for (var j = 0; j < this.Input2D.Y; j++)
                    {
                        for (var i = 0; i < this.Input2D.X; i++)
                        {
                            var pixel = 0.0;
                            var average = 0.0;
                            for (var y = -this.Size2D.Y / 2; y <= this.Size2D.Y / 2; y++)
                            {
                                for (var x = -this.Size2D.X / 2; x <= this.Size2D.X / 2; x++)
                                {
                                    if (0 > j + y || j + y >= this.Input2D.Y) continue;
                                    if (0 > i + x || i + x >= this.Input2D.X) continue;

                                    average += this.InputOutputData.Input[b][(i + x) + (j + y) * this.Input2D.X + ki];
                                    pixel += 1;
                                }
                            }
                            average /= pixel;

                            var sigma1 = 0.0;
                            var sigma2 = 0.0;
                            for (var y = -this.Size2D.Y / 2; y <= this.Size2D.Y / 2; y++)
                            {
                                for (var x = -this.Size2D.X / 2; x <= this.Size2D.X / 2; x++)
                                {
                                    if (0 > j + y || j + y >= this.Input2D.Y) continue;
                                    if (0 > i + x || i + x >= this.Input2D.X) continue;

                                    sigma1 = this.InputOutputData.Input[b][(i + x) + (j + y) * this.Input2D.X + ki] - average;
                                    sigma2 += sigma1 * sigma1;
                                }
                            }

                            sigma2 /= pixel;
                            sigma1 = Math.Sqrt(sigma2);
                            if (sigma1 < 1.0) sigma1 = 1.0;

                            var p = i + j * this.Input2D.X + ki;
                            this.InputOutputData.Output[b][p] = (this.InputOutputData.Input[b][p] - average) / sigma1;
                            this.GradData.Output[b][p] = 1.0;
                        }
                    }
                }
            });

            return new Tuple<double[][], double[][]>(this.InputOutputData.Output, this.GradData.Output);
        }

        public double[][] DeltaPropagation(double[][] delta)
        {
            this.DeltaData.SetInputData(delta);

            Parallel.For(0, this.DeltaData.Output.GetLength(0), b =>
            {
                for (var i = 0; i < this.DeltaData.Output[b].Length; i++)
                {
                    this.DeltaData.Output[b][i] = this.DeltaData.Input[b][i] * this.GradData.Input[b][i];
                }
            });

            return this.DeltaData.Output;
        }
        #endregion

        /// <summary>
        /// 内容を表す文字列を返します。
        /// </summary>
        /// <returns></returns>
        public override string ToString()
        {
            var sb = new StringBuilder();
            sb.AppendLine(nameof(Normalization));
            sb.AppendLine($"Input:({this.InputChannel},{this.Input2D.X},{this.Input2D.Y})");
            sb.AppendLine($"Filter:({this.Size2D.X},{this.Size2D.Y})");
            sb.Append($"Output:({this.OutputChannel},{this.Input2D.X},{this.Input2D.Y})");

            return sb.ToString();
        }
    }
}