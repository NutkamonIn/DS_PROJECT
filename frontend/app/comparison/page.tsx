'use client';

import Link from 'next/link';

export default function ComparisonPage() {
  return (
    <div className="min-h-screen bg-slate-50 p-8 font-sans text-slate-900">
      
      {/* Header */}
      <div className="max-w-6xl mx-auto flex justify-between items-center mb-10">
        <div>
          <h1 className="text-4xl font-extrabold text-indigo-900 tracking-tight">เปรียบเทียบโมเดล (Model Comparison)</h1>
          <p className="text-slate-500 mt-1 text-lg">วิเคราะห์เจาะลึก: Machine Learning vs Deep Learning</p>
        </div>
        <Link href="/" className="px-4 py-2 bg-white border border-slate-200 rounded-lg text-slate-600 hover:bg-slate-50 transition-colors font-semibold shadow-sm">
          กลับหน้าหลัก
        </Link>
      </div>

      <div className="max-w-7xl mx-auto space-y-12">

        {/* --- ตารางเปรียบเทียบรวม (Overview) --- */}
        <div className="bg-white rounded-3xl shadow-sm border border-slate-200 overflow-hidden">
          <div className="overflow-x-auto">
            <table className="w-full text-left border-collapse">
              <thead>
                <tr className="bg-indigo-50 border-b border-indigo-100">
                  <th className="p-4 text-indigo-900 font-extrabold text-base w-1/5">หัวข้อเปรียบเทียบ</th>
                  <th className="p-4 text-slate-700 font-bold text-base w-1/5">Logistic Regression</th>
                  <th className="p-4 text-slate-700 font-bold text-base w-1/5">Random Forest</th>
                  <th className="p-4 text-purple-700 font-bold text-base w-1/5">WangchanBERTa</th>
                  <th className="p-4 text-purple-700 font-bold text-base w-1/5">XLM-RoBERTa</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100 text-sm">
                <tr className="hover:bg-slate-50">
                  <td className="p-4 font-semibold text-slate-700">ประเภทโมเดล</td>
                  <td className="p-4">Linear Model (สมการเส้นตรง)</td>
                  <td className="p-4">Ensemble (Tree-based)</td>
                  <td className="p-4">Transformer (Encoder-only)</td>
                  <td className="p-4">Transformer (Multilingual)</td>
                </tr>
                <tr className="hover:bg-slate-50">
                  <td className="p-4 font-semibold text-slate-700">ความเร็วในการเทรน</td>
                  <td className="p-4 text-green-600 font-bold">เร็วมาก (วินาที)</td>
                  <td className="p-4 text-green-600 font-bold">เร็ว (นาที)</td>
                  <td className="p-4 text-red-500 font-bold">ช้ามาก (ชั่วโมง/GPU)</td>
                  <td className="p-4 text-red-500 font-bold">ช้ามาก (ชั่วโมง/GPU)</td>
                </tr>
                <tr className="hover:bg-slate-50">
                  <td className="p-4 font-semibold text-slate-700">ความเร็วตอนใช้งาน (Inference)</td>
                  <td className="p-4 text-green-600 font-bold">เร็วที่สุด (&lt; 10ms)</td>
                  <td className="p-4 text-green-600 font-bold">เร็ว (~20ms)</td>
                  <td className="p-4 text-yellow-600 font-bold">ปานกลาง (~100ms)</td>
                  <td className="p-4 text-yellow-600 font-bold">ปานกลาง (~120ms)</td>
                </tr>
                <tr className="hover:bg-slate-50">
                  <td className="p-4 font-semibold text-slate-700">ความเข้าใจบริบท</td>
                  <td className="p-4">ต่ำ (ดูแค่คำศัพท์ที่ปรากฏ)</td>
                  <td className="p-4">ปานกลาง (จับ Pattern ได้ดีกว่า)</td>
                  <td className="p-4 text-green-600 font-bold">สูงมาก (เข้าใจภาษาไทยลึกซึ้ง)</td>
                  <td className="p-4 text-green-600 font-bold">สูงมาก (รองรับหลายภาษา)</td>
                </tr>
                <tr className="hover:bg-slate-50">
                  <td className="p-4 font-semibold text-slate-700">ทรัพยากรที่ใช้</td>
                  <td className="p-4 text-green-600 font-bold">CPU ทั่วไป / RAM น้อย</td>
                  <td className="p-4 text-green-600 font-bold">CPU ทั่วไป / RAM ปานกลาง</td>
                  <td className="p-4 text-red-500 font-bold">High-end GPU / RAM สูง</td>
                  <td className="p-4 text-red-500 font-bold">High-end GPU / RAM สูงมาก</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>

        {/* --- รายละเอียดข้อจำกัด (Limitations) --- */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          
          {/* Logistic Regression */}
          <div className="bg-white p-6 rounded-2xl shadow-sm border-l-4 border-slate-400">
            <h3 className="text-lg font-bold text-slate-800 mb-3">1. Logistic Regression (Baseline)</h3>
            <ul className="list-disc list-outside ml-5 space-y-2 text-slate-600 text-sm">
              <li><strong>ข้อจำกัด:</strong> ไม่เข้าใจลำดับของคำ (Bag of Words) เช่น &quot;หมากัดคน&quot; กับ &quot;คนกัดหมา&quot; จะถูกมองว่าคล้ายกัน</li>
              <li><strong>ข้อจำกัด:</strong> แพ้ทางประโยคที่มีความซับซ้อน หรือประโยคประชดประชัน</li>
              <li><strong>ข้อดี:</strong> อธิบายผลลัพธ์ได้ง่ายมาก (ดู Weight ของแต่ละคำได้เลย)</li>
            </ul>
          </div>

          {/* Random Forest */}
          <div className="bg-white p-6 rounded-2xl shadow-sm border-l-4 border-slate-600">
            <h3 className="text-lg font-bold text-slate-800 mb-3">2. Random Forest</h3>
            <ul className="list-disc list-outside ml-5 space-y-2 text-slate-600 text-sm">
              <li><strong>ข้อจำกัด:</strong> โมเดลมีขนาดใหญ่ขึ้นตามจำนวนต้นไม้ (Trees) อาจกินแรมเยอะ</li>
              <li><strong>ข้อจำกัด:</strong> ไม่สามารถทำนายคำศัพท์ใหม่ๆ ที่ไม่เคยเจอใน Training set ได้ดีเท่า Deep Learning</li>
              <li><strong>ข้อดี:</strong> ทนทานต่อข้อมูลรบกวน (Noise) และไม่ต้องปรับแต่งค่าข้อมูลมากนัก</li>
            </ul>
          </div>

          {/* WangchanBERTa */}
          <div className="bg-white p-6 rounded-2xl shadow-sm border-l-4 border-purple-500">
            <h3 className="text-lg font-bold text-purple-800 mb-3">3. WangchanBERTa</h3>
            <ul className="list-disc list-outside ml-5 space-y-2 text-slate-600 text-sm">
              <li><strong>ข้อจำกัด:</strong> เทรนช้ามาก และต้องใช้ GPU เท่านั้นจึงจะทำงานได้ดี</li>
              <li><strong>ข้อจำกัด:</strong> จำกัดความยาวประโยคสูงสุด (512 tokens) ถ้าข่าวยาวเกินไปข้อมูลส่วนท้ายจะหายไป</li>
              <li><strong>ข้อดี:</strong> เป็นโมเดลที่ถูกเทรนมาเพื่อภาษาไทยโดยเฉพาะ เข้าใจสแลงและบริบทไทยได้ดีที่สุด</li>
            </ul>
          </div>

          {/* XLM-RoBERTa */}
          <div className="bg-white p-6 rounded-2xl shadow-sm border-l-4 border-pink-500">
            <h3 className="text-lg font-bold text-pink-800 mb-3">4. XLM-RoBERTa</h3>
            <ul className="list-disc list-outside ml-5 space-y-2 text-slate-600 text-sm">
              <li><strong>ข้อจำกัด:</strong> โมเดลมีขนาดใหญ่กว่า WangchanBERTa ทำให้ทำงานช้ากว่าเล็กน้อย</li>
              <li><strong>ข้อจำกัด:</strong> เนื่องจากต้องรองรับ 100 ภาษา อาจจะไม่เก่งภาษาไทยเฉพาะทางเท่า WangchanBERTa ในบางบริบท</li>
              <li><strong>ข้อดี:</strong> เหมาะมากถ้าในอนาคตต้องการรองรับข่าวภาษาอังกฤษหรือภาษาอื่นๆ ด้วย</li>
            </ul>
          </div>

        </div>

        {/* --- สรุปปิดท้าย --- */}
        <div className="bg-indigo-50 p-8 rounded-3xl text-center">
          <h3 className="text-xl font-bold text-indigo-900 mb-2">สรุปการเลือกใช้ (Recommendation)</h3>
          <p className="text-indigo-800 text-sm max-w-2xl mx-auto">
            หากต้องการความรวดเร็วและทรัพยากรจำกัด แนะนำให้ใช้ <strong>Logistic Regression</strong> 
            แต่หากต้องการความแม่นยำสูงสุดและมีทรัพยากรเพียงพอ <strong>WangchanBERTa</strong> คือตัวเลือกที่ดีที่สุดสำหรับภาษาไทย
          </p>
        </div>

      </div>
    </div>
  );
}