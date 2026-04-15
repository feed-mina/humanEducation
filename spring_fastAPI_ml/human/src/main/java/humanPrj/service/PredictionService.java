package humanPrj.service;

import humanPrj.dto.PredictionDTO;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;

import java.util.Arrays;
import java.util.List;
import org.springframework.http.converter.json.MappingJackson2HttpMessageConverter;

@Service
public class PredictionService {

    private static final String FASTAPI_URL = "http://localhost:8000/predict";

    public List<PredictionDTO> getPredictions() {
        RestTemplate restTemplate = new RestTemplate();

        // MappingJackson2HttpMessageConverter 추가
        restTemplate.getMessageConverters().add(new MappingJackson2HttpMessageConverter());

        try {
            // FastAPI에서 데이터를 가져옴
            PredictionDTO[] predictions = restTemplate.getForObject(FASTAPI_URL, PredictionDTO[].class);

            // 디버깅용 로그 출력
            if (predictions == null || predictions.length == 0) {
                System.out.println("No predictions received from FastAPI.");
                return List.of();
            }

            //System.out.println("Predictions received: " + Arrays.toString(predictions));

            return Arrays.asList(predictions);

        } catch (Exception e) {
            // 오류 처리
            System.err.println("Error fetching predictions: " + e.getMessage());
            e.printStackTrace();
            return List.of();
        }
    }
}

