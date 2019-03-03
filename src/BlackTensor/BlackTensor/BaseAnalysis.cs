using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace BlackTensor
{
    /// <summary>
    /// 基底の解析クラスです。
    /// </summary>
    public abstract class BaseAnalysis
    {
        #region プロパティ
        public int InputUnit { get; }
        public int OutputUnit { get; }
        public int BatchSample { get; }

        public InputOutpuData InputOutputData { get; private set; }
        public InputOutpuData DeltaData { get; private set; }
        public InputOutpuData GradData { get; private set; }
        #endregion

        #region 初期化
        /// <summary>
        /// 初期化します。
        /// </summary>
        protected BaseAnalysis()
        {
            this.Initialize();
        }
        /// <summary>
        /// 指定した値を使用して、初期化します。
        /// </summary>
        /// <param name="inputOutputUnit"></param>
        /// <param name="batchSample"></param>
        protected BaseAnalysis(int inputOutputUnit, int batchSample)
        {
            this.InputUnit = inputOutputUnit;
            this.OutputUnit = inputOutputUnit;
            this.BatchSample = batchSample;

            this.Initialize();
        }

        /// <summary>
        /// 指定した値を使用して、初期化します。
        /// </summary>
        /// <param name="inputUnit"></param>
        /// <param name="outputUnit"></param>
        /// <param name="batchSample"></param>
        /// <param name="inputOffset"></param>
        /// <param name="outputOffset"></param>
        protected BaseAnalysis(int inputUnit, int outputUnit, int batchSample, int inputOffset = 0, int outputOffset = 0)
        {
            this.InputUnit = inputUnit;
            this.OutputUnit = outputUnit;
            this.BatchSample = batchSample;

            this.Initialize(inputOffset, outputOffset);
        }

        private void Initialize(int inputOffset = 0, int outputOffset = 0)
        {
            var inputUnitOffset = this.InputUnit + inputOffset;
            var outputUnitOffset = this.OutputUnit + outputOffset;

            this.InputOutputData = new InputOutpuData(this.BatchSample, inputUnitOffset, outputUnitOffset);
            this.DeltaData = new InputOutpuData(this.BatchSample, outputUnitOffset, inputUnitOffset);
            this.GradData = new InputOutpuData(this.BatchSample, inputUnitOffset, outputUnitOffset);
        }
        #endregion


        public void SetInputGradData(double[][] inputData, double[][] gradData)
        {
            this.InputOutputData.SetInputData(inputData);
            this.GradData.SetInputData(gradData);
        }

        
    }
}
