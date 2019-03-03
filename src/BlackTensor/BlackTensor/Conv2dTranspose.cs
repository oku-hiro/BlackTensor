using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;

namespace BlackTensor
{
    public class Conv2DTranspose : BaseAnalysis
    {
        #region 定数
        private const int Upsampling = 2;
        private const double Beta1 = 0.9;
        private const double Beta2 = 0.999;
        private const double Gamma = 0.99;
        private readonly double _epsilon = Math.Pow(10.0, -8.0);
        #endregion

        #region プロパティ
        public int InputChannel { get; }
        public int FilterChannel { get; }
        public int OutputChannel { get; }

        public double Lr { get; }

        public Int2D Input2D { get; }
        public Int2D Filter2D { get; }
        public Int2D Output2D { get; }
        public Int2D Pad2D { get; }
        #endregion

        private readonly int _inputXy;
        private readonly int _outputXy;

        private readonly int _padXy;
        private readonly int _padUnit;

        private readonly double[][] _padding;
        private readonly double[][] _paddingDelta;
        private readonly double[][] _paddingGrad;
        private readonly double[] _filter;
        private readonly double[] _dFilter;
        private readonly double[] _bias;
        private readonly double[] _dBias;
        private readonly double[] _fm;
        private readonly double[] _fv;
        private readonly double[] _bm;
        private readonly double[] _bv;
        private readonly int[][] _connection;

        private double _b1, _b2;

        #region 初期化
        /// <inheritdoc />
        /// <summary>
        /// 初期化します。
        /// </summary>
        public Conv2DTranspose()
        {
            this._b1 = Beta1;
            this._b2 = Beta2;
        }
        /// <inheritdoc />
        /// <summary>
        /// 指定した値を使用して、初期化します。
        /// </summary>
        /// <param name="batchSample"></param>
        /// <param name="inputX"></param>
        /// <param name="inputY"></param>
        /// <param name="filterX"></param>
        /// <param name="filterY"></param>
        /// <param name="inputChannel"></param>
        /// <param name="filterChannel"></param>
        /// <param name="lr"></param>
        public Conv2DTranspose(int batchSample, int inputX, int inputY, int filterX, int filterY, int inputChannel, int filterChannel, double lr):
            base(inputX * inputY * inputChannel, (inputX * Upsampling) * (inputY * Upsampling) * filterChannel, batchSample)
        {
            this._b1 = Beta1;
            this._b2 = Beta2;

            this.InputChannel = inputChannel;
            this.FilterChannel = filterChannel;
            this.OutputChannel = filterChannel;
            this.Lr = lr;

            this.Input2D = new Int2D(inputX, inputY);
            this.Filter2D = new Int2D(filterX, filterY);
            this.Pad2D = new Int2D(Upsampling * this.Input2D.X, Upsampling * this.Input2D.Y);
            this.Output2D = new Int2D(this.Pad2D.X, this.Pad2D.Y);

            this._inputXy = this.Input2D.X * this.Input2D.Y;
            this._outputXy = this.Output2D.X * this.Output2D.Y;
            this._padXy = this.Pad2D.X * this.Pad2D.Y;
            var filterXy = this.Filter2D.X * this.Filter2D.Y;

            var filterElement = this.InputChannel * this.FilterChannel * filterXy;

            _padUnit = this.InputChannel * _padXy;

            _padding = new double[this.BatchSample][];
            _paddingDelta = new double[this.BatchSample][];
            _paddingGrad = new double[this.BatchSample][];

            for (var i = 0; i < this.BatchSample; i++)
            {
                _padding[i] = new double[_padUnit];
                _paddingDelta[i] = new double[_padUnit];
                _paddingGrad[i] = new double[_padUnit];
            }

            _filter = new double[filterElement];
            _dFilter = new double[filterElement];
            _bias = new double[this.FilterChannel];
            _dBias = new double[this.FilterChannel];

            _connection = new int[this.OutputUnit][];
            for (var i = 0; i < this._connection.GetLength(0); i++)
            {
                _connection[i] = new int[_padUnit];
                for (var j = 0; j < this._connection[i].Length; j++)
                {
                    _connection[i][j] = -1;
                }
            }

            _fm = new double[filterElement];
            _fv = new double[filterElement];
            _bm = new double[this.FilterChannel];
            _bv = new double[this.FilterChannel];

            for (var i = 0; i < filterElement; i++)
            {
                _fm[i] = 0.0;
                _fv[i] = 0.0;
            }

            for (var i = 0; i < this.FilterChannel; i++)
            {
                _bm[i] = 0.0;
                _bv[i] = 0.0;
            }

            var rnd = new Random();
            for (var i = 0; i < filterElement; i++)
            {
                _filter[i] = rnd.NextDouble();
                _filter[i] *= Math.Pow(10.0, -8.0);
            }

            for (var i = 0; i < this.FilterChannel; i++)
            {
                _bias[i] = 0.0;
            }

            for (var fk = 0; fk < this.FilterChannel; fk++)
            {
                for (var ik = 0; ik < this.InputChannel; ik++)
                {
                    for (var iy = 0; iy < this.Pad2D.Y; iy++)
                    {
                        for (var ix = 0; ix < this.Pad2D.X; ix++)
                        {
                            var f = ik * filterXy + fk * this.InputChannel * filterXy;
                            for (var fy = -this.Filter2D.Y / 2; fy <= this.Filter2D.Y / 2; fy++)
                            {
                                for (var fx = -this.Filter2D.X / 2; fx <= this.Filter2D.X / 2; fx++)
                                {
                                    if (0 <= ix + fx && ix + fx < this.Pad2D.X)
                                    {
                                        if (0 <= iy + fy && iy + fy < this.Pad2D.Y)
                                        {
                                            var p = ix + iy * this.Output2D.X + fk * _outputXy;         //output unit
                                            var q = (ix + fx) + (iy + fy) * this.Pad2D.X + ik * _padXy; //input unit
                                            _connection[p][q] = f;
                                        }
                                    }
                                    f++;
                                }
                            }
                        }
                    }
                }
            }
        }
        #endregion

        #region メソッド
        public Tuple<double[][], double[][]> Process(double[][] flow, double[][] grad)
        {
            this.SetInputGradData(flow, grad);

            Parallel.For(0, this.InputOutputData.Output.GetLength(0), b =>
            {
                for (var k = 0; k < this.InputChannel; k++)
                {
                    for (var j = 0; j < this.Input2D.Y; j++)
                    {
                        for (var i = 0; i < this.Input2D.X; i++)
                        {
                            _padding[b][Upsampling * (i + j * this.Pad2D.X) + k * _padXy] = this.InputOutputData.Input[b][i + j * this.Input2D.X + k * _inputXy];
                        }
                    }
                }

                for (var j = 0; j < this.InputOutputData.Output[b].Length; j++)
                {
                    this.InputOutputData.Output[b][j] = 0.0;
                    for (var i = 0; i < _padUnit; i++)
                    {
                        if (_connection[j][i] > -1)
                        {
                            this.InputOutputData.Output[b][j] += _filter[_connection[j][i]] * _padding[b][i];
                        }
                    }
                    this.InputOutputData.Output[b][j] += _bias[j / _outputXy];
                    this.GradData.Output[b][j] = 1.0;
                }
            });

            return new Tuple<double[][], double[][]>(this.InputOutputData.Output, this.GradData.Output);
        }

        public double[][] DeltaPropagation(double[][] delta)
        {
            this.DeltaData.SetInputData(delta);

            Parallel.For(0, this.DeltaData.Output.GetLength(0), b =>
            {
                for (var k = 0; k < this.InputChannel; k++)
                {
                    for (var j = 0; j < this.Input2D.Y; j++)
                    {
                        for (var i = 0; i < this.Input2D.X; i++)
                        {
                            _paddingGrad[b][Upsampling * (i + j * this.Pad2D.X) + k * _padXy] = this.GradData.Input[b][i + j * this.Input2D.X + k * _inputXy];
                        }
                    }
                }

                for (var j = 0; j < _padUnit; j++)
                {
                    _paddingDelta[b][j] = 0.0;
                    for (var i = 0; i < this.DeltaData.Input[b].Length; i++)
                    {
                        if (_connection[i][j] > -1)
                        {
                            _paddingDelta[b][j] += _filter[_connection[i][j]] * this.DeltaData.Input[b][i] * _paddingGrad[b][j];
                        }
                    }
                }

                for (var k = 0; k < this.InputChannel; k++)
                {
                    for (var j = 0; j < this.Input2D.Y; j++)
                    {
                        for (var i = 0; i < this.Input2D.X; i++)
                        {
                            this.DeltaData.Output[b][i + j * this.Input2D.X + k * _inputXy] = _paddingDelta[b][Upsampling * (i + j * this.Pad2D.X) + k * _padXy];
                        }
                    }
                }
            });

            return this.DeltaData.Output;
        }

        public void BackPropagation()
        {
            _b1 *= Beta1;
            _b2 *= Beta2;

            for (var i = 0; i < this._dFilter.Length; i++)
            {
                _dFilter[i] = 0.0;
            }

            for (var i = 0; i < this._dBias.Length; i++)
            {
                _dBias[i] = 0.0;
            }

            Parallel.For(0, this.DeltaData.Input.GetLength(0), k =>
            {
                for (var j = 0; j < this.DeltaData.Input[k].Length; j++)
                {
                    for (var i = 0; i < _padUnit; i++)
                    {
                        if (_connection[j][i] > -1)
                        {
                            _dFilter[_connection[j][i]] += this.DeltaData.Input[k][j] * _padding[k][i];
                        }
                    }
                }

                for (var j = 0; j < this._dBias.Length; j++)
                {
                    for (var i = 0; i < _outputXy; i++)
                    {
                        _dBias[j] += this.DeltaData.Input[k][i + j * _outputXy];
                    }
                }
            });
        }

        public void SGD()
        {
            for (var i = 0; i < this._filter.Length; i++)
            {
                _filter[i] -= Lr * _dFilter[i];
            }

            for (var i = 0; i < this._bias.Length; i++)
            {
                _bias[i] -= Lr * _dBias[i];
            }
        }

        public void ADAM()
        {
            for (var i = 0; i < this._filter.Length; i++)
            {
                _fm[i] = Beta1 * _fm[i] + (1.0 - Beta1) * _dFilter[i];
                _fv[i] = Beta2 * _fv[i] + (1.0 - Beta2) * _dFilter[i] * _dFilter[i];
                var m = _fm[i] / (1.0 - _b1);
                var v = _fv[i] / (1.0 - _b2);
                _filter[i] -= Lr * m / (Math.Sqrt(v) + _epsilon);
            }

            for (var i = 0; i < this._bias.Length; i++)
            {
                _bm[i] = Beta1 * _bm[i] + (1.0 - Beta1) * _dBias[i];
                _bv[i] = Beta2 * _bv[i] + (1.0 - Beta2) * _dBias[i] * _dBias[i];
                var m = _bm[i] / (1.0 - _b1);
                var v = _bv[i] / (1.0 - _b2);
                _bias[i] -= Lr * m / (Math.Sqrt(v) + _epsilon);
            }
        }

        public void RmsProp()
        {
            for (var i = 0; i < this._filter.Length; i++)
            {
                _fv[i] = Beta1 * _fv[i] + (1.0 - Gamma) * _dFilter[i] * _dFilter[i];
                _filter[i] -= Lr * _dFilter[i] / (Math.Sqrt(_fv[i]) + _epsilon);
            }

            for (var i = 0; i < this._bias.Length; i++)
            {
                _bv[i] = Beta2 * _bv[i] + (1.0 - Beta2) * _dBias[i] * _dBias[i];
                _bias[i] -= Lr * _dBias[i] / (Math.Sqrt(_bv[i]) + _epsilon);
            }
        }

        public void SaveParameter(int layer)
        {
            using (var sw1 = new StreamWriter("conv2d_transpose_filter" + (layer + 1)))
            {
                foreach (var item in _filter)
                {
                    sw1.WriteLine(item);
                }
            }

            using (var sw2 = new StreamWriter("conv2d_transpose_bias" + (layer + 1)))
            {
                foreach (var item in _bias)
                {
                    sw2.WriteLine(item);
                }
            }
        }
        #endregion

        /// <summary>
        /// 内容を表す文字列を返します。
        /// </summary>
        /// <returns></returns>
        public override string ToString()
        {
            var sb = new StringBuilder();
            sb.AppendLine(nameof(Conv2DTranspose));
            sb.AppendLine($"Input:({this.InputChannel},{this.Input2D.X},{this.Input2D.Y})");
            sb.AppendLine($"Filter:({this.FilterChannel},{this.Filter2D.X},{this.Filter2D.Y})");
            sb.Append($"Output:({this.OutputChannel},{this.Output2D.X},{this.Output2D.Y})");

            return sb.ToString();
        }
    }
}