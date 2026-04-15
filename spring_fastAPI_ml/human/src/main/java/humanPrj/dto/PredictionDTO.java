package humanPrj.dto;

import lombok.Data;
import com.fasterxml.jackson.annotation.JsonProperty;

@Data
public class PredictionDTO {

    @JsonProperty("Date") // JSON 필드 "Date"와 매핑
    private String date;

    @JsonProperty("Predicted_wattage") // JSON 필드 "Predicted_wattage"와 매핑
    private Double predictedWattage;
}
