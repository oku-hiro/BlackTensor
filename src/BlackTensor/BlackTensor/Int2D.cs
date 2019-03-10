using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace BlackTensor
{
    /// <summary>
    /// 2つの値を格納するクラスです。
    /// </summary>
    public class Int2D
    {
        #region プロパティ
        /// <summary>
        /// X の値を取得・設定します。
        /// </summary>
        public int X { get; set; }
        /// <summary>
        /// Y の値を取得・設定します。
        /// </summary>
        public int Y { get; set; }
        /// <summary>
        /// X と Y の合計値を取得します。
        /// </summary>
        public int Sum => this.X + this.Y;
        /// <summary>
        /// X と Y を掛けた値を取得します。
        /// </summary>
        public int Xy => this.X * this.Y;
        #endregion

        #region 初期化
        /// <summary>
        /// 初期化します。
        /// </summary>
        public Int2D() { }

        /// <summary>
        /// 指定した値を使用して、初期化します。
        /// </summary>
        /// <param name="x"></param>
        /// <param name="y"></param>
        public Int2D(int x, int y)
        {
            this.X = x;
            this.Y = y;
        }
        #endregion
    }
}
