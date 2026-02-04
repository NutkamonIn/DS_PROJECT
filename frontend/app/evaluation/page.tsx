'use client';

import Link from 'next/link';

export default function EvaluationPage() {
  return (
    <div className="min-h-screen bg-slate-50 p-8 font-sans text-slate-900">
      
      {/* Header */}
      <div className="max-w-6xl mx-auto flex justify-between items-center mb-10">
        <div>
          <h1 className="text-4xl font-extrabold text-indigo-900 tracking-tight">Model Evaluation</h1>
          <p className="text-slate-500 mt-1 text-lg">Detailed performance analysis using Confusion Matrix</p>
        </div>
        <Link href="/" className="px-4 py-2 bg-white border border-slate-200 rounded-lg text-slate-600 hover:bg-slate-50 transition-colors font-semibold shadow-sm">
          Back to Arena
        </Link>
      </div>

      <div className="max-w-7xl mx-auto space-y-12">
        
        {/* Info Section */}
        <div className="space-y-6">
          <div className="bg-indigo-900 text-white p-8 rounded-3xl shadow-xl">
            <h2 className="text-2xl font-bold mb-4">Confusion Matrix คืออะไร?</h2>
            <p className="text-indigo-100 leading-relaxed opacity-90">
              Confusion Matrix คือตารางที่ใช้สำหรับอธิบายประสิทธิภาพของโมเดลประเภทการจำแนกกลุ่ม (Classification Model) 
              โดยจะแสดงจำนวนการทำนายที่ &quot;ถูกต้อง&quot; และ &quot;ผิดพลาด&quot; แบ่งตามแต่ละคลาส 
              ข้อมูลในแนวทแยงมุมจะแทนจำนวนข้อมูลที่โมเดลทำนายได้ถูกต้อง (ทำนายได้ตรงกับค่าจริง) 
              ในขณะที่ข้อมูลที่อยู่นอกแนวทแยงคือจำนวนข้อมูลที่โมเดลทำนายผิดพลาดไปเป็นคลาสอื่นนั่นเอง
            </p>
          </div>

          <div className="bg-white p-8 rounded-3xl shadow-sm border border-indigo-50">
            <h3 className="text-xl font-bold text-indigo-900 mb-4">เกณฑ์การอ่านผลลัพธ์ (How to Read)</h3>
            <ul className="space-y-3 text-slate-600">
              <li className="flex items-start gap-3">
                <div className="w-12 h-6 flex items-center justify-center bg-green-100 text-green-700 rounded text-[10px] font-bold tracking-wider">PASS</div>
                <div>
                  <span className="font-bold text-slate-800">แนวทแยงมุมต้อง &quot;สีเข้ม/เลขเยอะ&quot;</span>
                  <p className="text-sm mt-1">ตัวเลขในช่องแนวทแยง (จากซ้ายบนลงขวาล่าง) คือจำนวนที่โมเดล <span className="text-green-600 font-semibold">ทายถูก</span> ยิ่งเลขเยอะหรือสีเข้ม ยิ่งแสดงว่าโมเดลแม่นยำ</p>
                </div>
              </li>
              <li className="flex items-start gap-3">
                <div className="w-12 h-6 flex items-center justify-center bg-red-100 text-red-700 rounded text-[10px] font-bold tracking-wider">FAIL</div>
                <div>
                  <span className="font-bold text-slate-800">นอกแนวทแยงต้อง &quot;สีอ่อน/เลขน้อย&quot;</span>
                  <p className="text-sm mt-1">ตัวเลขที่อยู่นอกแนวทแยงคือจำนวนที่โมเดล <span className="text-red-500 font-semibold">ทายผิด</span> (สับสนไปเป็นคลาสอื่น) ยิ่งเลขน้อยหรือสีจาง ยิ่งดี</p>
                </div>
              </li>
            </ul>
          </div>
        </div>

        {/* Images Grid Section (2x2) */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          
          {/* 1. WangchanBERTa */}
          <div className="bg-white p-6 rounded-2xl shadow-md border-t-4 border-indigo-500 flex flex-col items-center text-center">
            <h3 className="text-lg font-bold text-indigo-900 mb-4 uppercase tracking-wide">WangchanBERTa</h3>
            <div className="bg-slate-50 rounded-xl overflow-hidden w-full aspect-[4/3] flex items-center justify-center border border-slate-100">
              <img 
                src="http://localhost:8000/evaluation/image/bert" 
                alt="WangchanBERTa Matrix"
                className="max-w-full max-h-full object-contain"
              />
            </div>
            <p className="mt-4 text-xs text-slate-400">Deep Learning (Thai Specific)</p>
          </div>

          {/* 2. XLM-RoBERTa */}
          <div className="bg-white p-6 rounded-2xl shadow-md border-t-4 border-fuchsia-500 flex flex-col items-center text-center">
            <h3 className="text-lg font-bold text-fuchsia-900 mb-4 uppercase tracking-wide">XLM-RoBERTa</h3>
            <div className="bg-slate-50 rounded-xl overflow-hidden w-full aspect-[4/3] flex items-center justify-center border border-slate-100">
              <img 
                src="http://localhost:8000/evaluation/image/xlmr" 
                alt="XLM-RoBERTa Matrix"
                className="max-w-full max-h-full object-contain"
              />
            </div>
            <p className="mt-4 text-xs text-slate-400">Deep Learning (Multilingual)</p>
          </div>

          {/* 3. Logistic Regression */}
          <div className="bg-white p-6 rounded-2xl shadow-md border-t-4 border-teal-500 flex flex-col items-center text-center">
            <h3 className="text-lg font-bold text-teal-900 mb-4 uppercase tracking-wide">Logistic Regression</h3>
            <div className="bg-slate-50 rounded-xl overflow-hidden w-full aspect-[4/3] flex items-center justify-center border border-slate-100">
              <img 
                src="http://localhost:8000/evaluation/image/logreg" 
                alt="Logistic Regression Matrix"
                className="max-w-full max-h-full object-contain"
              />
            </div>
            <p className="mt-4 text-xs text-slate-400">Baseline (Linear Model)</p>
          </div>

          {/* 4. Random Forest */}
          <div className="bg-white p-6 rounded-2xl shadow-md border-t-4 border-amber-500 flex flex-col items-center text-center">
            <h3 className="text-lg font-bold text-amber-900 mb-4 uppercase tracking-wide">Random Forest</h3>
            <div className="bg-slate-50 rounded-xl overflow-hidden w-full aspect-[4/3] flex items-center justify-center border border-slate-100">
              <img 
                src="http://localhost:8000/evaluation/image/rf" 
                alt="Random Forest Matrix"
                className="max-w-full max-h-full object-contain"
              />
            </div>
            <p className="mt-4 text-xs text-slate-400">Ensemble (Tree-based)</p>
          </div>

        </div>

        {/* Footer */}
        <div className="text-center pb-10">
          <p className="text-slate-400 text-sm">
            All data generated via scripts/benchmark_all.py
          </p>
        </div>

      </div>
    </div>
  );
}