
import React from 'react';

// 예시 데이터. 실제 애플리케이션에서는 API를 통해 받아오거나, 날짜별로 다른 문장을 보여주는 로직이 필요합니다.
const sentenceData = {
  japanese: '今日は天気が良いですね。',
  korean: '오늘은 날씨가 좋네요.',
  breakdown: [
    {
      word: '今日',
      reading: 'きょう (kyou)',
      meaning: '오늘',
      partOfSpeech: '명사',
    },
    {
      word: 'は',
      reading: 'わ (wa)',
      meaning: '~은/는',
      partOfSpeech: '조사',
      explanation: '문장의 주제를 나타냅니다.',
    },
    {
      word: '天気',
      reading: 'てんき (tenki)',
      meaning: '날씨',
      partOfSpeech: '명사',
    },
    {
      word: 'が',
      reading: 'が (ga)',
      meaning: '~이/가',
      partOfSpeech: '조사',
      explanation: '문장의 주어를 나타냅니다.',
    },
    {
      word: '良い',
      reading: 'いい (ii)',
      meaning: '좋다',
      partOfSpeech: 'い-형용사',
    },
    {
      word: 'ですね',
      reading: 'ですね (desu ne)',
      meaning: '~네요, ~군요',
      partOfSpeech: '표현',
      explanation: '상대방의 동의를 구하거나, 감탄을 나타낼 때 사용합니다.',
    },
  ],
};

const DailyJapaneseSentence = () => {
  return (
    <div className="p-6 bg-white border border-gray-200 rounded-lg shadow-md">
      <h2 className="text-2xl font-bold mb-4 text-gray-800 border-b pb-2">
        오늘의 일본어 문장
      </h2>
      
      {/* 일본어 문장 및 번역 */}
      <div className="mb-6 text-center">
        <p className="text-3xl font-semibold text-blue-600 mb-2">
          {sentenceData.japanese}
        </p>
        <p className="text-lg text-gray-600">
          {sentenceData.korean}
        </p>
      </div>

      {/* 문장 분석 */}
      <div>
        <h3 className="text-xl font-semibold mb-3 text-gray-700">
          문장 분석
        </h3>
        <div className="space-y-3">
          {sentenceData.breakdown.map((item, index) => (
            <div key={index} className="p-3 bg-gray-50 rounded-md border-l-4 border-blue-500">
              <div className="flex items-baseline space-x-3">
                <span className="text-lg font-bold text-gray-900">{item.word}</span>
                <span className="text-sm font-mono text-gray-500">{item.reading}</span>
                <span className="text-sm px-2 py-0.5 bg-blue-100 text-blue-800 rounded-full">
                  {item.partOfSpeech}
                </span>
              </div>
              <p className="mt-1 text-md text-gray-700">- {item.meaning}</p>
              {item.explanation && (
                <p className="mt-1 text-sm text-gray-600 pl-2 border-l-2 border-gray-300 ml-1">
                  {item.explanation}
                </p>
              )}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default DailyJapaneseSentence;
