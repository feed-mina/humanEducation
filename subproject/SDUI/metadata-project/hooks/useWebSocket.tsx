import { useCallback, useRef } from 'react';
import { Client } from '@stomp/stompjs';

export const useWebSocket = () => {
    // 1. 웹소켓 클라이언트 객체를 유지하기 위한 참조 변수야.
    const client = useRef<Client | null>(null);

    // 2. 서버로 메시지를 보내는 핵심 함수야.
    const sendMessage = useCallback((destination: string, body: any) => {
        // 클라이언트가 연결되어 있고 활성화된 상태인지 확인해.
        if (client.current && client.current.active) {
            client.current.publish({
                destination,
                body: JSON.stringify(body),
            });
        } else {
            console.warn("웹소켓이 연결되지 않았어. 메시지를 보낼 수 없어.");
        }
    }, []);

    // 3. 실제 연결 로직은 여기서 관리하게 될 거야. (추후 구현)

    return { sendMessage };
};