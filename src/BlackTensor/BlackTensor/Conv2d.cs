using System;
using System.Collections.Generic;
using System.Data.SqlTypes;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;

namespace BlackTensor
{
    public class Conv2D : BaseAnalysis
    {
        #region 定数
        private const double Beta1 = 0.9;
        private const double Beta2 = 0.999;
        private const double Gamma = 0.99;
        private readonly double _eps = Math.Pow(10.0, -8.0);
        #endregion

        #region プロパティ
        public int InputChannel { get; }
        public int FilterChannel { get; }
        public int OutputChannel { get; }

        public int Stride { get; }
        public double Lr { get; }

        public Int2D Input2D { get; }
        public Int2D Filter2D { get; }
        public Int2D Output2D { get; }
        #endregion

        //private readonly int _outputXy;

        private readonly int _filterElement;

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
        public Conv2D()
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
        /// <param name="stride"></param>
        /// <param name="inputChannel"></param>
        /// <param name="filterChannel"></param>
        /// <param name="lr"></param>
        public Conv2D(int batchSample, int inputX, int inputY, int filterX, int filterY, int stride, int inputChannel, int filterChannel, double lr) : 
            base(inputX * inputY * inputChannel, (inputX / stride) * (inputY / stride) * filterChannel, batchSample)
        {
            this._b1 = Beta1;
            this._b2 = Beta2;

            this.Stride = stride;
            this.InputChannel = inputChannel;
            this.FilterChannel = filterChannel;
            this.OutputChannel = filterChannel;
            this.Lr = lr;

            this.Input2D = new Int2D(inputX, inputY);
            this.Filter2D = new Int2D(filterX, filterY);
            this.Output2D = new Int2D(this.Input2D.X / stride, this.Input2D.Y / stride);
            

            this._filterElement = InputChannel * FilterChannel * this.Filter2D.Xy;


            this._filter = new double[this._filterElement];
            this._dFilter = new double[this._filterElement];


            _bias = new double[FilterChannel];
            _dBias = new double[FilterChannel];

            this._connection = new int[this.OutputUnit][];
            for (var i = 0; i < this._connection.GetLength(0); i++)
            {
                _connection[i] = new int[_filterElement];
            }

            _fm = new double[_filterElement];
            _fv = new double[_filterElement];
            _bm = new double[FilterChannel];
            _bv = new double[FilterChannel];

            for (var i = 0; i < _filterElement; i++)
            {
                _fm[i] = 0.0;
                _fv[i] = 0.0;
            }

            for (var i = 0; i < FilterChannel; i++)
            {
                _bm[i] = 0.0;
                _bv[i] = 0.0;
            }

            var rnd = new Random();
            for (var i = 0; i < this._filter.Length; i++)
            {
                this._filter[i] = rnd.NextDouble();
                //filter[i] *= Math.Pow(10.0, -8.0);
            }

            for (var i = 0; i < FilterChannel; i++)
            {
                _bias[i] = 0.0;
            }

            for (var j = 0; j < this.OutputUnit; j++)
            {
                for (var i = 0; i < _filterElement; i++)
                {
                    _connection[j][i] = -1;
                }
            }

            for (var fk = 0; fk < FilterChannel; fk++)
            {
                for (var ik = 0; ik < InputChannel; ik++)
                {
                    for (var iy = 0; iy < this.Input2D.Y; iy += Stride)
                    {
                        for (var ix = 0; ix < this.Input2D.X; ix += Stride)
                        {
                            var f = ik * this.Filter2D.Xy + fk * InputChannel * this.Filter2D.Xy;
                            for (var fy = -this.Filter2D.Y / 2; fy <= this.Filter2D.Y / 2; fy++)
                            {
                                for (var fx = -this.Filter2D.X / 2; fx <= this.Filter2D.X / 2; fx++)
                                {
                                    if (0 <= ix + fx && ix + fx < this.Input2D.X)
                                    {
                                        if (0 <= iy + fy && iy + fy < this.Input2D.Y)
                                        {
                                            var p = ix / Stride + iy / Stride * this.Output2D.X + fk * this.Output2D.Xy;      //output unit
                                            _connection[p][f] = (ix + fx) + (iy + fy) * this.Input2D.X + ik * this.Input2D.Xy; //input unit
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

            var sw = new Stopwatch();
            sw.Start();

            Parallel.For(0, this.InputOutputData.Output.GetLength(0), b =>
            {
                var inputData = this.InputOutputData.Input[b];
                for (var j = 0; j < this.InputOutputData.Output[b].Length; j++)
                {
                    this.InputOutputData.Output[b][j] = 0.0;
                    for (var i = 0; i < this._filter.Length; i++)
                    {
                        var c = _connection[j][i];
                        if (c > -1)
                        {
                            //this.InputOutputData.Output[b][j] += _filter[i] * this.InputOutputData.Input[b][c];
                            this.InputOutputData.Output[b][j] += _filter[i] * inputData[c];
                        }
                    }
                    this.InputOutputData.Output[b][j] += _bias[j / this.Output2D.Xy];
                    //Console.WriteLine(output[b][j]);
                    this.GradData.Output[b][j] = 1.0;
                }
            });

            //for (var b = 0; b < this.InputOutputData.Output.GetLength(0); b++)
            //{
            //    for (var j = 0; j < this.InputOutputData.Output[b].Length; j++)
            //    {
            //        this.InputOutputData.Output[b][j] = 0.0;
            //        for (var i = 0; i < this._filter.Length; i++)
            //        {
            //            if (_connection[j][i] > -1)
            //            {
            //                this.InputOutputData.Output[b][j] += _filter[i] * this.InputOutputData.Input[b][_connection[j][i]];
            //            }
            //        }
            //        this.InputOutputData.Output[b][j] += _bias[j / _outputXy];
            //        //Console.WriteLine(output[b][j]);
            //        this.GradData.Output[b][j] = 1.0;
            //    }
            //}

            sw.Stop();
            Debug.WriteLine($"{nameof(Conv2D)}.{nameof(this.Process)}：{sw.ElapsedMilliseconds}[ms]");

            return new Tuple<double[][], double[][]>(this.InputOutputData.Output, this.GradData.Output); 
        }

        public double[][] DeltaPropagation(double[][] delta)
        {
            this.DeltaData.SetInputData(delta);

            var sw = new Stopwatch();
            sw.Start();

            Parallel.For(0, this.DeltaData.Output.GetLength(0), b =>
            {
                for (var i = 0; i < this.DeltaData.Output[b].Length; i++)
                {
                    this.DeltaData.Output[b][i] = 0.0;
                }

                var deltaData = this.DeltaData.Input[b];
                var gradData = this.GradData.Input[b];

                for (var j = 0; j < this._filter.Length; j++)
                {
                    var filterData = _filter[j];
                    for (var i = 0; i < this.DeltaData.Input[b].Length; i++)
                    {
                        var c = _connection[i][j];
                        if (c > -1)
                        {
                            //this.DeltaData.Output[b][c] += _filter[j] * this.DeltaData.Input[b][i] * this.GradData.Input[b][c];
                            this.DeltaData.Output[b][c] += filterData * deltaData[i] * gradData[c];
                        }
                        //if (_connection[i][j] > -1)
                        //{
                        //    this.DeltaData.Output[b][_connection[i][j]] += _filter[j] * this.DeltaData.Input[b][i] * this.GradData.Input[b][_connection[i][j]];
                        //}
                    }
                }
            });

            //for (var b = 0; b < this.DeltaData.Output.GetLength(0); b++)
            //{
            //    for (var i = 0; i < this.DeltaData.Output[b].Length; i++)
            //    {
            //        this.DeltaData.Output[b][i] = 0.0;
            //    }

            //    for (var j = 0; j < this._filter.Length; j++)
            //    {
            //        for (var i = 0; i < this.DeltaData.Input[b].Length; i++)
            //        {
            //            if (_connection[i][j] > -1)
            //            {
            //                this.DeltaData.Output[b][_connection[i][j]] += _filter[j] * this.DeltaData.Input[b][i] * this.GradData.Input[b][_connection[i][j]];
            //            }
            //        }
            //    }
            //}

            sw.Stop();
            Debug.WriteLine($"{nameof(Conv2D)}.{nameof(this.DeltaPropagation)}：{sw.ElapsedMilliseconds}[ms]");

            return this.DeltaData.Output;
        }

        public void BackPropagation()
        {
            var sw = new Stopwatch();
            sw.Start();

            _b1 *= Beta1;
            _b2 *= Beta2;

            //for (var i = 0; i < this._dFilter.Length; i++)
            //{
            //    _dFilter[i] = 0.0;
            //}

            //for (var i = 0; i < this._dBias.Length; i++)
            //{
            //    _dBias[i] = 0.0;
            //}

            var ddFilter = new double[this.BatchSample][];
            var ddBias = new double[this.BatchSample][];
            for (var i = 0; i < this.BatchSample; i++)
            {
                ddFilter[i] = new double[this._dFilter.Length];
                ddBias[i] = new double[this._dBias.Length];
            }

            Parallel.For(0, this.BatchSample, b =>
            {
                var deltaData = this.DeltaData.Input[b];
                var inputData = this.InputOutputData.Input[b];

                for (var j = 0; j < this._dFilter.Length; j++)
                {
                    for (var i = 0; i < this.DeltaData.Input[b].Length; i++)
                    {
                        var c = _connection[i][j];
                        if (c > -1)
                        {
                            //ddFilter[b][j] += this.DeltaData.Input[b][i] * this.InputOutputData.Input[b][c];
                            ddFilter[b][j] += deltaData[i] * inputData[c];
                        }
                    }
                }

                for (var j = 0; j < this.FilterChannel; j++)
                {
                    var value = j * this.Output2D.Xy;
                    for (var i = 0; i < this.Output2D.Xy; i++)
                    {
                        //ddBias[b][j] += this.DeltaData.Input[b][i + j * this.Output2D.Xy];
                        ddBias[b][j] += deltaData[i + value];
                    }
                }
            });

            for (var b = 0; b < this.BatchSample; b++)
            {
                for (var i = 0; i < this._dFilter.Length; i++)
                {
                    this._dFilter[i] += ddFilter[b][i];
                }

                for (var i = 0; i < this._dBias.Length; i++)
                {
                    this._dBias[i] += ddBias[b][i];
                }
            }

            //for (var b = 0; b < this.BatchSample; b++)
            //{
            //    var b1 = b;
            //    Parallel.For(0, this._dFilter.Length, j =>
            //    {
            //        for (var i = 0; i < this.DeltaData.Input[b1].Length; i++)
            //        {
            //            if (_connection[i][j] > -1)
            //            {
            //                _dFilter[j] += this.DeltaData.Input[b1][i] * this.InputOutputData.Input[b1][_connection[i][j]];
            //            }
            //        }
            //    });

            //    var b2 = b;
            //    Parallel.For(0, FilterChannel, j =>
            //    {
            //        for (var i = 0; i < _outputXy; i++)
            //        {
            //            _dBias[j] += this.DeltaData.Input[b2][i + j * _outputXy];
            //        }
            //    });
            //}

            //for (var b = 0; b < this.BatchSample; b++)
            //{

            //    for (var j = 0; j < this._dFilter.Length; j++)
            //    {
            //        for (var i = 0; i < this.DeltaData.Input[b].Length; i++)
            //        {
            //            if (_connection[i][j] > -1)
            //            {
            //                _dFilter[j] += this.DeltaData.Input[b][i] * this.InputOutputData.Input[b][_connection[i][j]];
            //            }
            //        }
            //    }

            //    for (var j = 0; j < FilterChannel; j++)
            //    {
            //        for (var i = 0; i < _outputXy; i++)
            //        {
            //            _dBias[j] += this.DeltaData.Input[b][i + j * _outputXy];
            //        }
            //    }
            //}

            sw.Stop();
            Debug.WriteLine($"{nameof(Conv2D)}.{nameof(this.BackPropagation)}：{sw.ElapsedMilliseconds}[ms]");
        }

        public void SGD()
        {
            for (var i = 0; i < this._filter.Length; i++)
            {
                this._filter[i] -= Lr * this._dFilter[i];
            }

            for (var i = 0; i < FilterChannel; i++)
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
                this._filter[i] -= Lr * m / (Math.Sqrt(v) + _eps);
            }

            for (var i = 0; i < FilterChannel; i++)
            {
                _bm[i] = Beta1 * _bm[i] + (1.0 - Beta1) * _dBias[i];
                _bv[i] = Beta2 * _bv[i] + (1.0 - Beta2) * _dBias[i] * _dBias[i];
                var m = _bm[i] / (1.0 - _b1);
                var v = _bv[i] / (1.0 - _b2);
                _bias[i] -= Lr * m / (Math.Sqrt(v) + _eps);
            }
        }

        public void RmsProp()
        {
            for (var i = 0; i < this._filter.Length; i++)
            {
                _fv[i] = Beta1 * _fv[i] + (1.0 - Gamma) * _dFilter[i] * _dFilter[i];
                this._filter[i] -= Lr * _dFilter[i] / (Math.Sqrt(_fv[i]) + _eps);
            }

            for (var i = 0; i < this.FilterChannel; i++)
            {
                _bv[i] = Beta2 * _bv[i] + (1.0 - Beta2) * _dBias[i] * _dBias[i];
                _bias[i] -= Lr * _dBias[i] / (Math.Sqrt(_bv[i]) + _eps);
            }
        }

        public void SaveParameter(int layer)
        {
            using (var sw1 = new StreamWriter("conv2d_filter" + (layer + 1)))
            {
                for (var i = 0; i < _filterElement; i++)
                {
                    sw1.WriteLine(_filter[i]);
                }
            }
            using (var sw2 = new StreamWriter("conv2d_bias" + (layer + 1)))
            {
                for (var i = 0; i < FilterChannel; i++)
                {
                    sw2.WriteLine(_bias[i]);
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
            sb.AppendLine(nameof(Conv2D));
            sb.AppendLine($"Input:({this.InputChannel},{this.Input2D.X},{this.Input2D.Y})");
            sb.AppendLine($"Filter:({this.FilterChannel},{this.Filter2D.X},{this.Filter2D.Y})");
            sb.Append($"Output:({this.OutputChannel},{this.Output2D.X},{this.Output2D.Y})");

            return sb.ToString();
        }
    }
}