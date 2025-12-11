import numpy as np
import onnxruntime as ort


class LightGlueRunner:
    def __init__(
        self,
        lightglue_path: str,
        extractor_path=None,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    ):
        self.extractor = (
            ort.InferenceSession(extractor_path, providers=providers)
            if extractor_path is not None
            else None
        )
        self.lightglue = ort.InferenceSession(lightglue_path, providers=providers)

        # Check for invalid models.
        lightglue_inputs = [i.name for i in self.lightglue.get_inputs()]
        if self.extractor is not None and "image0" in lightglue_inputs:
            raise TypeError(
                f"The specified LightGlue model at {lightglue_path} is end-to-end. Please do not pass the extractor_path argument."
            )
        elif self.extractor is None and "image0" not in lightglue_inputs:
            raise TypeError(
                f"The specified LightGlue model at {lightglue_path} is not end-to-end. Please pass the extractor_path argument."
            )

    def run(self, image0: np.ndarray, image1: np.ndarray, scales0, scales1):
        if self.extractor is None:
            kpts0, kpts1, matches0, matches1, mscores0, mscores1 = self.lightglue.run(
                None,
                {
                    "image0": image0,
                    "image1": image1,
                },
            )
            m_kpts0, m_kpts1 = self.post_process(
                kpts0, kpts1, matches0, scales0, scales1
            )
            return m_kpts0, m_kpts1
        else:
            kpts0, scores0, desc0 = self.extractor.run(None, {"image": image0})
            kpts1, scores1, desc1 = self.extractor.run(None, {"image": image1})

            kpts0 = self.normalize_keypoints(
                        kpts0, image0.shape[2], image0.shape[3]
                    ) 
            kpts1 = self.normalize_keypoints(
                        kpts1, image0.shape[2], image0.shape[3]
                    ) 
            
            kpts0 = np.expand_dims(kpts0, axis=-1)
            kpts1 = np.expand_dims(kpts1, axis=-1)
            desc0 = np.expand_dims(desc0, axis=-1)
            desc1 = np.expand_dims(desc1, axis=-1)
            
            print("kpts0 shape:", kpts0.shape)
            print("kpts0 min/max:", kpts0.min(), kpts0.max())
            print("kpts1 shape:", kpts1.shape)
            print("kpts1 min/max:", kpts1.min(), kpts1.max())
            print("desc0 shape:", desc0.shape)
            print("desc0 min/max:", desc0.min(), desc0.max())
            print("desc1 shape:", desc1.shape)
            print("desc1 min/max:", desc1.min(), desc1.max())
            print("scores0 range:", scores0.min(), scores0.max())
            print("scores1 range:", scores1.min(), scores1.max())


            matches0, matches1, mscores0, mscores1 = self.lightglue.run(
                None,
                {
                    "kpts0": kpts0,
                    "kpts1": kpts1,
                    "desc0": desc0 ,
                    "desc1": desc1 ,
                },
            )
            
            print("LightGlue matches0 range:",matches0.min(), matches0.max(), "mscores0 range:", mscores0.min(), mscores0.max())
            print("LightGlue matches1 range:",matches1.min(), matches1.max(), "mscores1 range:", mscores1.min(), mscores1.max())
            
            m_kpts0, m_kpts1 = self.post_process(
                kpts0, kpts1, matches0, scales0, scales1
            )
            return m_kpts0, m_kpts1

    def normalize_keypoints(
        self,
        kpts: np.ndarray,
        h: int,
        w: int,
    ) -> np.ndarray:
        size = np.array([w, h])
        shift = size / 2
        scale = size.max() / 2
        kpts = (kpts - shift) / scale
        return kpts.astype(np.float32)

    def post_process(self, kpts0, kpts1, matches0, scales0, scales1):
        kpts0 = (kpts0 + 0.5) / scales0 - 0.5
        kpts1 = (kpts1 + 0.5) / scales1 - 0.5
        # create match indices
        valid = matches0[0] > -1
        matches = np.stack([np.where(valid)[0], matches0[0][valid]], -1)
        m_kpts0, m_kpts1 = kpts0[0][matches[..., 0]], kpts1[0][matches[..., 1]]
        return m_kpts0, m_kpts1
