package humanPrj.controller;
import humanPrj.dto.PredictionDTO;
import humanPrj.service.PredictionService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.client.RestTemplate;
import java.util.List;

@Controller
public class PredictionController {

    @Autowired
    private PredictionService predictionService;

    @GetMapping("/prediction")
    public String getPredictions(Model model) {
        List<PredictionDTO> predictions = predictionService.getPredictions();

        // 데이터가 제대로 전달되는지 확인
        /*if (predictions.isEmpty()) {
            System.out.println("No predictions available to display.");
        } else {
            System.out.println("Predictions: " + predictions);
        }*/

        model.addAttribute("predictions", predictions);
        return "prediction";  // prediction.jsp로 반환
    }
}