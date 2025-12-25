# Strict-GS-M290
Strict GS algorithm based on symmetric time reversal on 4f system

Demo video: https://www.bilibili.com/video/BV1QJBUBEELE/?spm_id_from=333.1387.homepage.video_card.click&vd_source=b154583a3cd22b39e08fccf0ef5e396e


This video visualizes the real-time phase retrieval results of a standard strict Gerchberg–Saxton (GS) algorithm based on symmetric time reversal (i.e., backward propagation), together with several intermediate computational states of particular interest during the iterative process.

As in all practical applications of the GS algorithm, a forward simulation of the optical system must be accompanied by a corresponding backward simulation, where each step in the backward propagation is mathematically and programmatically the exact inverse of its counterpart in the forward propagation. The video demonstrates such a standard GS algorithm developed for a non-typical 4f optical system.

The experimental results shown in the video are obtained from the high-energy beam shaping system of the M290 laser–metal powder selective laser melting (SLM) additive manufacturing machine within the EU InShaPe project. This beam shaping system constitutes a non-typical 4f configuration, in which the lenses are arranged sequentially as one concave lens, followed by a convex lens, and then another convex lens. As illustrated in the video, a solid circular Gaussian beam is shaped into a distorted serrated beam, a distorted square beam, or a distorted hollow annular beam. The standard GS algorithm can conveniently determine a phase-only computer-generated hologram (CGH) mask that reshapes the input Gaussian beam into an arbitrary target beam profile.

Experimental results indicate that although the beam shaping accuracy achieved by the standard GS algorithm is significantly higher than that of the pseudo-GS algorithm based on virtual-image alignment, the similarity between the CGH mask recovered by the standard GS algorithm and the ground-truth CGH mask is substantially lower than that obtained by the pseudo-GS algorithm. While in most practical applications the similarity between the recovered CGH mask and the ground truth is of limited concern, this observation is nevertheless noteworthy from the perspective of phase retrieval as a purely physical inverse problem: a pseudo algorithm relying on approximate inverse propagation yields higher phase reconstruction accuracy than the standard algorithm employing a strict inverse simulation. This counterintuitive phenomenon merits further in-depth investigation in future work.

The standard GS algorithm presented in this video is implemented as part of the direct GS unrolling model proposed in my recent journal paper published in Advanced Photonics Nexus (SPIE):

“Efficient Gerchberg–Saxton algorithm deep unrolling for phase retrieval with complex forward path.”
