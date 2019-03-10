using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace BlackTensor
{
    public class Pooling : BaseAnalysis
    {
        #region プロパティ
        public Int2D Input2D { get; }
        public Int2D Size2D { get; }
        public Int2D Output2D { get; }

        public int InputChannel { get; }
        public int OutputChannel { get; }
        #endregion


        private readonly int _inputXy;
        private readonly int _outputXy;
        private readonly int[][] _w;

        #region 初期化
        /// <inheritdoc />
        /// <summary>
        /// 初期化します。
        /// </summary>
        public Pooling() {}
        /// <inheritdoc />
        /// <summary>
        /// 指定した値を使用して、初期化します。
        /// </summary>
        /// <param name="batchSample"></param>
        /// <param name="inputX"></param>
        /// <param name="inputY"></param>
        /// <param name="sizeX"></param>
        /// <param name="sizeY"></param>
        /// <param name="inputChannel"></param>
        public Pooling(int batchSample, int inputX, int inputY, int sizeX, int sizeY, int inputChannel) : 
            base(inputX * inputY * inputChannel, (inputX /sizeX) * (inputY / sizeY) * inputChannel, batchSample)
        {
            this.Input2D = new Int2D(inputX, inputY);
            this.Size2D = new Int2D(sizeX, sizeY);
            this.Output2D = new Int2D(this.Input2D.X / this.Size2D.X, this.Input2D.Y / this.Size2D.Y);
            this.InputChannel = inputChannel;
            this.OutputChannel = inputChannel;

            _inputXy = this.Input2D.X * this.Input2D.Y;
            _outputXy = this.Output2D.X * this.Output2D.Y;

            this._w = new int[batchSample][];

            for (var i = 0; i < batchSample; i++)
            {
                this._w[i] = new int[this.OutputUnit];
            }
        }
        #endregion

        #region メソッド
        public Tuple<double[][], double[][]> Process(double[][] flow, double[][] grad)
        {
            this.SetInputGradData(flow, grad);

            for (var b = 0; b < this.BatchSample; b++)
            {
                for (var i = 0; i < this.OutputUnit; i++)
                {
                    this.GradData.Output[b][i] = 0.0;
                    //this_grad[b][i] = 0.0;
                    _w[b][i] = -1;
                }
           
                for (var k = 0; k < InputChannel; k++)
                {
                    var ki = k * _inputXy;
                    var kp = k * _outputXy;

                    for (var j = 0; j < this.Output2D.Y; j++)
                    {
                        for (var i = 0; i < this.Output2D.X; i++)
                        {
                            var startX = i * this.Size2D.X;
                            var startY = j * this.Size2D.Y;
                            var maxX = startX;
                            var maxY = startY;

                            var maximum = this.InputOutputData.Input[b][0];
                            for (var y = 0; y < this.Size2D.Y; y++)
                            {
                                for (var x = 0; x < this.Size2D.X; x++)
                                {
                                    if (!(maximum < this.InputOutputData.Input[b][(startX + x) + (startY + y) * this.Input2D.X + ki])) continue;

                                    maxX = startX + x;
                                    maxY = startY + y;
                                    maximum = this.InputOutputData.Input[b][maxX + maxY * this.Input2D.X + ki];
                                }
                            }

                            var p = i + j * this.Output2D.X + kp;
                            _w[b][p] = maxX + maxY * this.Input2D.X + ki;
                            this.InputOutputData.Output[b][p] = maximum;
                            this.GradData.Output[b][p] = 1.0;
                            //this_grad[b][p] = 1.0;
                        }
                    }
                }
            }

            return new Tuple<double[][], double[][]>(this.InputOutputData.Output, this.GradData.Output);
        }

        public double[][] DeltaPropagation(double[][] delta)
        {
            this.DeltaData.SetInputData(delta);

            for (var b = 0; b < this.InputOutputData.Output.GetLength(0); b++)
            {
                for (var i = 0; i < this.InputOutputData.Output[b].Length; i++)
                {
                    this.DeltaData.Output[b][i] = 0.0;
                }

                for (var i = 0; i < this.InputOutputData.Output[b].Length; i++)
                {
                    if (_w[b][i] > -1)
                    {
                        this.DeltaData.Output[b][_w[b][i]] += this.DeltaData.Input[b][i] * this.GradData.Input[b][_w[b][i]];
                    }
                }
            }

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
            sb.AppendLine(nameof(Pooling));
            sb.AppendLine($"Input:({this.InputChannel},{this.Input2D.X},{this.Input2D.Y})");
            sb.AppendLine($"Filter:({this.Size2D.X},{this.Size2D.Y})");
            sb.Append($"Output:({this.OutputChannel},{this.Output2D.X},{this.Output2D.Y})");

            return sb.ToString();
        }
    }
}