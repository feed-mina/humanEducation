package com.domain.demo_backend.domain.tutorial.dto;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

@Getter
@Setter
@NoArgsConstructor
public class TutorialSaveRequest {

    @JsonProperty("component_id")
    private String componentId;

    @JsonProperty("component_type")
    private String componentType;

    @JsonProperty("label_text")
    private String labelText;

    @JsonProperty("sort_order")
    private Integer sortOrder;

    @JsonProperty("css_class")
    private String cssClass;
}