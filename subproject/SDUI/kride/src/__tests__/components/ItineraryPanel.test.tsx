import { render, screen, fireEvent } from "@testing-library/react";
import ItineraryPanel from "@/components/kride/ItineraryPanel";

const MOCK_DAY_PLAN = {
  morning: { places: [{ name: "경복궁", description: "조선 정궁" }] },
  afternoon: { places: [{ name: "남산서울타워" }] },
};

describe("ItineraryPanel - Focus duration별 날짜 구조", () => {
  it("당일치기(day)는 1일치 패널만 렌더링된다", () => {
    render(
      <ItineraryPanel
        id="test"
        meta={{}}
        data={{ duration: "day", itinerary: [MOCK_DAY_PLAN] }}
      />
    );
    // "당일" 헤더가 있어야 한다
    expect(screen.getByText("당일")).toBeInTheDocument();
    // "Day 2"는 없어야 한다
    expect(screen.queryByText("Day 2")).toBeNull();
  });

  it("1박2일(onenight)은 2일치 패널이 렌더링된다", () => {
    render(
      <ItineraryPanel
        id="test"
        meta={{}}
        data={{ duration: "onenight", itinerary: [MOCK_DAY_PLAN, MOCK_DAY_PLAN] }}
      />
    );
    expect(screen.getByText("Day 1")).toBeInTheDocument();
    expect(screen.getByText("Day 2")).toBeInTheDocument();
    expect(screen.queryByText("Day 3")).toBeNull();
  });

  it("2박3일(twonight)은 3일치 패널이 렌더링된다", () => {
    render(
      <ItineraryPanel
        id="test"
        meta={{}}
        data={{
          duration: "twonight",
          itinerary: [MOCK_DAY_PLAN, MOCK_DAY_PLAN, MOCK_DAY_PLAN],
        }}
      />
    );
    expect(screen.getByText("Day 1")).toBeInTheDocument();
    expect(screen.getByText("Day 2")).toBeInTheDocument();
    expect(screen.getByText("Day 3")).toBeInTheDocument();
  });

  it("오전/오후 아코디언 헤더가 각 Day마다 렌더링된다", () => {
    render(
      <ItineraryPanel
        id="test"
        meta={{}}
        data={{ duration: "day", itinerary: [MOCK_DAY_PLAN] }}
      />
    );
    expect(screen.getByText("오전")).toBeInTheDocument();
    expect(screen.getByText("오후")).toBeInTheDocument();
  });

  it("오전 헤더를 클릭하면 장소가 펼쳐진다", () => {
    render(
      <ItineraryPanel
        id="test"
        meta={{}}
        data={{ duration: "day", itinerary: [MOCK_DAY_PLAN] }}
      />
    );
    fireEvent.click(screen.getByText("오전"));
    expect(screen.getByText("경복궁")).toBeInTheDocument();
  });

  it("일정이 없을 때 '일정이 없습니다' 메시지가 표시된다", () => {
    const emptyPlan = {
      morning: { places: [] },
      afternoon: { places: [] },
    };
    render(
      <ItineraryPanel
        id="test"
        meta={{}}
        data={{ duration: "day", itinerary: [emptyPlan] }}
      />
    );
    fireEvent.click(screen.getByText("오전"));
    expect(screen.getAllByText("일정이 없습니다").length).toBeGreaterThan(0);
  });
});
