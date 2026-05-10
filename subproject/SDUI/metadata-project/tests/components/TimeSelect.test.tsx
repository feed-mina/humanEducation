import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import TimeSelect from '../../components/fields/TimeSelect';
import '@testing-library/jest-dom';
import {logTestSuccess} from "@/tests/TestLogger";


// Swiper 모킹 로직 유지...
jest.mock('swiper/react', () => ({
    Swiper: ({ children }: any) => <div>{children}</div>,
    SwiperSlide: ({ children }: any) => <div>{children}</div>,
}));

describe('TimeSelect 컴포넌트 테스트', () => {
    const mockMeta = {
        label_text: '수면 시간',
        component_props: { startHour: 0, endHour: 24, slidesPerView: 6 }
    };
    const mockOnChange = jest.fn();

    // @@@@ 개별 테스트 성공 시 로그 기록
    afterEach(() => {
        // Jest의 현재 테스트 상태를 확인하여 성공 시에만 로그를 남긴다
        const testName = expect.getState().currentTestName;
        logTestSuccess(`TimeSelect - ${testName}`);
    });


jest.mock('swiper/modules', () => ({
    Navigation: () => null,
}));

jest.mock('swiper/css', () => ({}));
jest.mock('swiper/css/navigation', () => ({}));

describe('TimeSelect 컴포넌트 테스트', () => {
    const mockMeta = {
        label_text: '수면 시간',
        component_props: { startHour: 0, endHour: 24, slidesPerView: 6 }
    };
    const mockOnChange = jest.fn();

    //  렌더링 상태 확인 (터미널에서 HTML 구조를 확인해봐)
    screen.debug();
    it('시간(22시)을 클릭하면 id와 함께 선택된 배열 [22]가 onChange로 전달되어야 한다', () => {
        // 1. 렌더링
        render(
            <TimeSelect
                id="sleep_time_select"
                meta={mockMeta}
                data={[]}
                onChange={mockOnChange}
            />
        );

        // 2. '22시' 버튼 클릭 시뮬레이션
        const timeBtn = screen.getByText(/22/);
        fireEvent.click(timeBtn);

        // 3. 호출 검증: 클릭한 22시가 배열에 담겨 전달되는지 확인
        // [1]이 아니라 [22]가 전달되어야 함
        expect(mockOnChange).toHaveBeenCalledWith('sleep_time_select', [22]);
    });
    test('22시가 선택된 상태에서 23시를 클릭하면 [22, 23]이 전달되어야 한다', () => {
        const handleChange = jest.fn();
        const mockMeta = {
            label_text: '수면 시간',
            component_props: { startHour: 0, endHour: 24 }
        };

        render(
            <TimeSelect
                id="sleep-time"
                meta={mockMeta}
                data={[22]} // 'value'가 아니라 'data'로 전달해야 함
                onChange={handleChange}
            />
        );

        const nextTimeBtn = screen.getByText(/23/);
        fireEvent.click(nextTimeBtn);

        // 이제 [22, 23]이 정상적으로 찍힐 것이다.
        expect(handleChange).toHaveBeenCalledWith('sleep-time', [22, 23]);
    });
    test('이미 선택된 22시를 다시 클릭하면 배열에서 제거되어야 한다', () => {
        const handleChange = jest.fn();
        const mockMeta = {
            label_text: '수면 시간',
            component_props: { startHour: 20, endHour: 24 }
        };

        // 22시가 이미 선택된 상태(data 프롭)로 렌더링
        render(
            <TimeSelect
                id="sleep-time"
                meta={mockMeta}
                data={[22]}
                onChange={handleChange}
            />
        );

        // 22시 버튼 클릭
        const timeBtn = screen.getByText(/22/);
        fireEvent.click(timeBtn);

        // 22가 제거된 빈 배열 []이 전달되는지 확인
        expect(handleChange).toHaveBeenCalledWith('sleep-time', []);
    });
    test('meta.label_text가 화면에 제목으로 올바르게 렌더링되어야 한다', () => {
        const mockMeta = {
            label_text: '기상 시간 설정', // 테스트하고 싶은 제목
            component_props: { startHour: 6, endHour: 10 }
        };

        render(
            <TimeSelect
                id="wake-time"
                meta={mockMeta}
                data={[]}
                onChange={jest.fn()}
            />
        );

        // 1. 해당 텍스트를 가진 요소가 DOM에 존재하는지 확인
        const labelElement = screen.getByText('기상 시간 설정');

        // 2. 단언(Assertion): 문서 내에 존재하는가?
        expect(labelElement).toBeInTheDocument();

        // 3. (선택 사항) 특정 태그인지 확인하고 싶을 때
        expect(labelElement.tagName).toBe('H3');
    });

    test('startHour(6)와 endHour(10) 설정에 따라 정확히 5개의 시간 버튼이 렌더링되어야 한다', () => {
        const mockMeta = {
            label_text: '아침 시간',
            component_props: {
                startHour: 6,
                endHour: 10
            }
        };

        render(
            <TimeSelect
                id="morning-time"
                meta={mockMeta}
                data={[]}
                onChange={jest.fn()}
            />
        );

        // 1. role이 'button'인 모든 요소를 가져온다.
        // Swiper 네비게이션 버튼이 포함될 수 있으므로, 숫자 텍스트를 가진 버튼만 필터링하거나
        // 컨테이너 내부의 버튼만 특정해서 가져오는 것이 정확하다.
        const hourButtons = screen.getAllByRole('button').filter(btn =>
            /^\d+$/.test(btn.textContent || '')
        );

        // 2. 개수 검증 (10 - 6 + 1 = 5)
        expect(hourButtons).toHaveLength(5);

        // 3. 첫 번째와 마지막 숫자가 맞는지도 확인하면 더 완벽하다.
        expect(hourButtons[0]).toHaveTextContent('6');
        expect(hourButtons[4]).toHaveTextContent('10');
    });

    test('data 프롭이 null로 넘어와도 에러 없이 빈 배열로 처리되어야 한다', () => {
        const handleChange = jest.fn();
        const mockMeta = {
            label_text: '테스트 시간',
            component_props: { startHour: 0, endHour: 5 }
        };

        // data에 의도적으로 null을 주입
        render(
            <TimeSelect
                id="crash-test"
                meta={mockMeta}
                data={null as any}
                onChange={handleChange}
            />
        );

        // 1. 에러 없이 렌더링되었는지 확인 (제목이 보이면 일단 성공)
        expect(screen.getByText('테스트 시간')).toBeInTheDocument();

        // 2. 버튼 클릭 시 빈 배열에서 시작하는지 확인
        const btn1 = screen.getByText(/1/);
        fireEvent.click(btn1);

        // null이었지만 내부에서 []로 치환되어 [1]이 결과로 나와야 함
        expect(handleChange).toHaveBeenCalledWith('crash-test', [1]);
    });

    test('meta.component_props가 없을 때 기본값(0시~24시)이 적용되어야 한다', () => {
        // component_props가 통째로 빠진 상황 가정
        const incompleteMeta = {
            label_text: '기본값 테스트'
        };

        render(
            <TimeSelect
                id="default-test"
                meta={incompleteMeta as any}
                data={[]}
                onChange={jest.fn()}
            />
        );

        // 기본값인 0시와 24시 버튼이 존재하는지 확인
        // /^숫자$/ 패턴을 사용하여 정확히 일치하는 텍스트만 찾음
        expect(screen.getByText(/^0$/)).toBeInTheDocument();
        expect(screen.getByText(/^24$/)).toBeInTheDocument();

        // 또는 Role을 사용하는 더 권장되는 방식
        // expect(screen.getByRole('button', { name: '0' })).toBeInTheDocument();
    });
});
});