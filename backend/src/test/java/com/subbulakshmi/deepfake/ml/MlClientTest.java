package com.subbulakshmi.deepfake.ml;

import com.subbulakshmi.deepfake.config.MlProperties;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.springframework.http.HttpMethod;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.test.web.client.MockRestServiceServer;
import org.springframework.web.client.RestClient;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.hamcrest.Matchers.containsString;
import static org.springframework.test.web.client.match.MockRestRequestMatchers.content;
import static org.springframework.test.web.client.match.MockRestRequestMatchers.method;
import static org.springframework.test.web.client.match.MockRestRequestMatchers.requestTo;
import static org.springframework.test.web.client.response.MockRestResponseCreators.withServerError;
import static org.springframework.test.web.client.response.MockRestResponseCreators.withStatus;
import static org.springframework.test.web.client.response.MockRestResponseCreators.withSuccess;

/**
 * Verifies the wire format against the ML service (multipart request, snake_case JSON response)
 * without a real HTTP server.
 */
class MlClientTest {

    private static final String PREDICTION_JSON = """
            {
              "label": "FAKE",
              "confidence": 0.9134,
              "fake_probability": 0.9134,
              "real_probability": 0.0866,
              "frames_analyzed": 20,
              "frames_total": 300,
              "faces_detected": 19,
              "fps": 29.97,
              "duration_s": 10.0,
              "processing_ms": 4321,
              "file_name": "clip.mp4",
              "size_bytes": 12,
              "model": {
                "backbone": "resnext50_32x4d",
                "recurrent": "lstm",
                "profile": "default",
                "weights": null,
                "trained": false,
                "device": "cpu"
              }
            }
            """;

    private MockRestServiceServer server;
    private MlClient client;

    @TempDir
    Path tempDir;

    @BeforeEach
    void setUp() {
        RestClient.Builder builder = RestClient.builder().baseUrl("http://ml-service:8000");
        server = MockRestServiceServer.bindTo(builder).build();
        RestClient restClient = builder.build();
        client = new MlClient(restClient, restClient);
    }

    @Test
    void predictPostsMultipartAndParsesSnakeCaseJson() throws Exception {
        Path video = tempDir.resolve("stored.mp4");
        Files.write(video, "fake-video-bytes".getBytes());

        server.expect(requestTo("http://ml-service:8000/predict"))
                .andExpect(method(HttpMethod.POST))
                .andExpect(content().contentTypeCompatibleWith(MediaType.MULTIPART_FORM_DATA))
                .andExpect(content().string(containsString("name=\"file\"")))
                .andExpect(content().string(containsString("filename=\"clip.mp4\"")))
                .andExpect(content().string(containsString("Content-Type: video/mp4")))
                .andExpect(content().string(containsString("fake-video-bytes")))
                .andRespond(withSuccess(PREDICTION_JSON, MediaType.APPLICATION_JSON));

        MlPrediction p = client.predict(video, "clip.mp4", "video/mp4");

        server.verify();
        assertThat(p.label()).isEqualTo("FAKE");
        assertThat(p.confidence()).isEqualTo(0.9134);
        assertThat(p.fakeProbability()).isEqualTo(0.9134);
        assertThat(p.realProbability()).isEqualTo(0.0866);
        assertThat(p.framesAnalyzed()).isEqualTo(20);
        assertThat(p.facesDetected()).isEqualTo(19);
        assertThat(p.processingMs()).isEqualTo(4321L);
        assertThat(p.model()).isEqualTo(new MlModelInfo("resnext50_32x4d", "lstm", null, false));
    }

    @Test
    void predictWrapsHttpErrorsInMlServiceException() throws Exception {
        Path video = tempDir.resolve("stored.mp4");
        Files.write(video, new byte[]{1, 2, 3});

        server.expect(requestTo("http://ml-service:8000/predict"))
                .andRespond(withStatus(HttpStatus.BAD_REQUEST)
                        .contentType(MediaType.APPLICATION_JSON)
                        .body("{\"detail\":\"no face detected\"}"));

        assertThatThrownBy(() -> client.predict(video, "clip.mp4", "video/mp4"))
                .isInstanceOf(MlServiceException.class)
                .hasMessageContaining("HTTP 400")
                .hasMessageContaining("no face detected");
    }

    @Test
    void predictWrapsServerErrors() throws Exception {
        Path video = tempDir.resolve("stored.mp4");
        Files.write(video, new byte[]{1});

        server.expect(requestTo("http://ml-service:8000/predict")).andRespond(withServerError());

        assertThatThrownBy(() -> client.predict(video, "clip.mp4", null))
                .isInstanceOf(MlServiceException.class)
                .hasMessageContaining("HTTP 500");
    }

    @Test
    void healthReturnsBodyOnSuccess() {
        server.expect(requestTo("http://ml-service:8000/health"))
                .andExpect(method(HttpMethod.GET))
                .andRespond(withSuccess("{\"status\":\"ok\",\"model_loaded\":true}", MediaType.APPLICATION_JSON));

        Map<String, Object> health = client.health();

        assertThat(health).containsEntry("status", "ok").containsEntry("model_loaded", true);
    }

    @Test
    void healthReturnsDownOnFailure() {
        server.expect(requestTo("http://ml-service:8000/health")).andRespond(withServerError());

        Map<String, Object> health = client.health();

        assertThat(health).containsEntry("status", "down").containsKey("error");
    }

    @Test
    void healthReturnsDownWhenServiceIsUnreachable() {
        MlProperties props = new MlProperties("http://127.0.0.1:1", 300, 300);
        MlClient real = new MlClient(RestClient.builder(), props);

        Map<String, Object> health = real.health();

        assertThat(health).containsEntry("status", "down").containsKey("error");
    }
}
