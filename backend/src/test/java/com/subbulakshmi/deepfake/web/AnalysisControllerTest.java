package com.subbulakshmi.deepfake.web;

import com.subbulakshmi.deepfake.domain.Verdict;
import com.subbulakshmi.deepfake.domain.VideoAnalysis;
import com.subbulakshmi.deepfake.domain.VideoAnalysisRepository;
import com.subbulakshmi.deepfake.ml.MlClient;
import com.subbulakshmi.deepfake.ml.MlModelInfo;
import com.subbulakshmi.deepfake.ml.MlPrediction;
import com.subbulakshmi.deepfake.ml.MlServiceException;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.AutoConfigureMockMvc;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.test.mock.mockito.MockBean;
import org.springframework.mock.web.MockMultipartFile;
import org.springframework.test.context.ActiveProfiles;
import org.springframework.test.context.DynamicPropertyRegistry;
import org.springframework.test.context.DynamicPropertySource;
import org.springframework.test.web.servlet.MockMvc;
import org.springframework.test.web.servlet.MvcResult;

import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;
import java.util.List;
import java.util.Map;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.when;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.delete;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.multipart;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.content;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

/**
 * End-to-end tests of the REST API against H2 with the ML client mocked out.
 */
@SpringBootTest
@AutoConfigureMockMvc
@ActiveProfiles("test")
class AnalysisControllerTest {

    @TempDir
    static Path storageDir;

    @DynamicPropertySource
    static void storageProperties(DynamicPropertyRegistry registry) {
        registry.add("app.storage.dir", () -> storageDir.toString());
    }

    @Autowired
    private MockMvc mockMvc;

    @Autowired
    private VideoAnalysisRepository repository;

    @Autowired
    private ObjectMapper objectMapper;

    @MockBean
    private MlClient mlClient;

    private static final byte[] VIDEO_BYTES = "not really a video but good enough".getBytes();

    @BeforeEach
    void cleanDatabase() {
        repository.deleteAll();
    }

    private static MlPrediction fakePrediction() {
        return new MlPrediction("fake", 0.93, 0.93, 0.07, 32, 30, 1234L,
                new MlModelInfo("resnext50_32x4d", "lstm", "model.pt", true));
    }

    @Test
    void uploadingVideoReturns201AndPersistsRow() throws Exception {
        when(mlClient.predict(any(Path.class), eq("clip.mp4"), anyString())).thenReturn(fakePrediction());

        MockMultipartFile file = new MockMultipartFile("file", "clip.mp4", "video/mp4", VIDEO_BYTES);

        MvcResult result = mockMvc.perform(multipart("/api/analyses").file(file))
                .andExpect(status().isCreated())
                .andExpect(content().contentTypeCompatibleWith("application/json"))
                .andExpect(jsonPath("$.id").isNumber())
                .andExpect(jsonPath("$.fileName").value("clip.mp4"))
                .andExpect(jsonPath("$.sizeBytes").value(VIDEO_BYTES.length))
                .andExpect(jsonPath("$.verdict").value("FAKE"))
                .andExpect(jsonPath("$.confidence").value(0.93))
                .andExpect(jsonPath("$.fakeProbability").value(0.93))
                .andExpect(jsonPath("$.framesAnalyzed").value(32))
                .andExpect(jsonPath("$.facesDetected").value(30))
                .andExpect(jsonPath("$.processingMs").value(1234))
                .andExpect(jsonPath("$.modelName").value("resnext50_32x4d+lstm · model.pt"))
                .andExpect(jsonPath("$.errorMessage").doesNotExist())
                .andExpect(jsonPath("$.createdAt").isString())
                .andReturn();

        JsonNode json = objectMapper.readTree(result.getResponse().getContentAsString());
        long id = json.get("id").asLong();

        VideoAnalysis row = repository.findById(id).orElseThrow();
        assertThat(row.getVerdict()).isEqualTo(Verdict.FAKE);
        assertThat(row.getOriginalFileName()).isEqualTo("clip.mp4");
        assertThat(row.getContentType()).isEqualTo("video/mp4");
        assertThat(row.getStoredFileName()).endsWith(".mp4").isNotEqualTo("clip.mp4");
        assertThat(row.getCreatedAt()).isNotNull();

        Path storedFile = storageDir.resolve(row.getStoredFileName());
        assertThat(storedFile).exists();
        assertThat(Files.readAllBytes(storedFile)).isEqualTo(VIDEO_BYTES);
    }

    @Test
    void uploadingNonVideoReturns400() throws Exception {
        MockMultipartFile file = new MockMultipartFile("file", "notes.txt", "text/plain", "hello".getBytes());

        mockMvc.perform(multipart("/api/analyses").file(file))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.status").value(400))
                .andExpect(jsonPath("$.error").value("Bad Request"))
                .andExpect(jsonPath("$.message").isString())
                .andExpect(jsonPath("$.timestamp").isString());

        assertThat(repository.count()).isZero();
    }

    @Test
    void uploadingEmptyFileReturns400() throws Exception {
        MockMultipartFile file = new MockMultipartFile("file", "empty.mp4", "video/mp4", new byte[0]);

        mockMvc.perform(multipart("/api/analyses").file(file))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.status").value(400));
    }

    @Test
    void missingFilePartReturns400NotFoundNot500() throws Exception {
        mockMvc.perform(multipart("/api/analyses"))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.status").value(400));
    }

    @Test
    void historyReturnsItemsArray() throws Exception {
        when(mlClient.predict(any(Path.class), anyString(), anyString())).thenReturn(fakePrediction());
        mockMvc.perform(multipart("/api/analyses")
                        .file(new MockMultipartFile("file", "a.mp4", "video/mp4", VIDEO_BYTES)))
                .andExpect(status().isCreated());

        mockMvc.perform(get("/api/analyses").param("page", "0").param("size", "20"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.items").isArray())
                .andExpect(jsonPath("$.items.length()").value(1))
                .andExpect(jsonPath("$.items[0].fileName").value("a.mp4"))
                .andExpect(jsonPath("$.page").value(0))
                .andExpect(jsonPath("$.size").value(20))
                .andExpect(jsonPath("$.total").value(1));
    }

    @Test
    void statsReturnsCountersAndNullAverageWhenEmpty() throws Exception {
        mockMvc.perform(get("/api/analyses/stats"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.total").value(0))
                .andExpect(jsonPath("$.fake").value(0))
                .andExpect(jsonPath("$.real").value(0))
                .andExpect(jsonPath("$.errors").value(0))
                .andExpect(jsonPath("$.avgConfidence").value((Object) null));
    }

    @Test
    void statsAverageIgnoresErrorRows() throws Exception {
        repository.save(row("x.mp4", Verdict.FAKE, 0.9));
        repository.save(row("y.mp4", Verdict.REAL, 0.7));
        repository.save(row("z.mp4", Verdict.ERROR, null));

        mockMvc.perform(get("/api/analyses/stats"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.total").value(3))
                .andExpect(jsonPath("$.fake").value(1))
                .andExpect(jsonPath("$.real").value(1))
                .andExpect(jsonPath("$.errors").value(1))
                .andExpect(jsonPath("$.avgConfidence").value(0.8));
    }

    @Test
    void mlFailureReturns502AndPersistsErrorRow() throws Exception {
        when(mlClient.predict(any(Path.class), anyString(), anyString()))
                .thenThrow(new MlServiceException("ML service unreachable: Connection refused"));

        MockMultipartFile file = new MockMultipartFile("file", "broken.mov", "video/quicktime", VIDEO_BYTES);

        mockMvc.perform(multipart("/api/analyses").file(file))
                .andExpect(status().isBadGateway())
                .andExpect(content().contentTypeCompatibleWith("application/json"))
                .andExpect(jsonPath("$.status").value(502))
                .andExpect(jsonPath("$.error").value("Bad Gateway"))
                .andExpect(jsonPath("$.message").value("ML service unreachable: Connection refused"));

        List<VideoAnalysis> rows = repository.findAll();
        assertThat(rows).hasSize(1);
        VideoAnalysis row = rows.get(0);
        assertThat(row.getVerdict()).isEqualTo(Verdict.ERROR);
        assertThat(row.getOriginalFileName()).isEqualTo("broken.mov");
        assertThat(row.getErrorMessage()).contains("Connection refused");
        assertThat(row.getConfidence()).isNull();
        assertThat(repository.countByVerdict(Verdict.ERROR)).isEqualTo(1);
    }

    @Test
    void getByIdReturnsRowAnd404WhenMissing() throws Exception {
        VideoAnalysis saved = repository.save(row("one.webm", Verdict.REAL, 0.88));

        mockMvc.perform(get("/api/analyses/" + saved.getId()))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.id").value(saved.getId()))
                .andExpect(jsonPath("$.verdict").value("REAL"));

        mockMvc.perform(get("/api/analyses/999999"))
                .andExpect(status().isNotFound())
                .andExpect(jsonPath("$.status").value(404))
                .andExpect(jsonPath("$.error").value("Not Found"));
    }

    @Test
    void deleteRemovesRowAndStoredFile() throws Exception {
        when(mlClient.predict(any(Path.class), anyString(), anyString())).thenReturn(fakePrediction());
        MvcResult created = mockMvc.perform(multipart("/api/analyses")
                        .file(new MockMultipartFile("file", "gone.mkv", "video/x-matroska", VIDEO_BYTES)))
                .andExpect(status().isCreated())
                .andReturn();
        long id = objectMapper.readTree(created.getResponse().getContentAsString()).get("id").asLong();
        Path stored = storageDir.resolve(repository.findById(id).orElseThrow().getStoredFileName());
        assertThat(stored).exists();

        mockMvc.perform(delete("/api/analyses/" + id)).andExpect(status().isNoContent());

        assertThat(repository.findById(id)).isEmpty();
        assertThat(stored).doesNotExist();

        mockMvc.perform(delete("/api/analyses/" + id)).andExpect(status().isNotFound());
    }

    @Test
    void healthCombinesBackendAndMlStatus() throws Exception {
        when(mlClient.health()).thenReturn(Map.of("status", "ok", "trained", true));

        mockMvc.perform(get("/api/health"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.backend").value("ok"))
                .andExpect(jsonPath("$.mlService.status").value("ok"))
                .andExpect(jsonPath("$.mlService.trained").value(true));
    }

    @Test
    void unknownRouteIsJson404NotA500() throws Exception {
        mockMvc.perform(get("/api/does-not-exist"))
                .andExpect(status().isNotFound())
                .andExpect(jsonPath("$.status").value(404));
    }

    private static VideoAnalysis row(String name, Verdict verdict, Double confidence) {
        VideoAnalysis a = new VideoAnalysis();
        a.setOriginalFileName(name);
        a.setStoredFileName("stored-" + name);
        a.setContentType("video/mp4");
        a.setSizeBytes(10);
        a.setVerdict(verdict);
        a.setConfidence(confidence);
        a.setCreatedAt(Instant.now());
        return a;
    }
}
