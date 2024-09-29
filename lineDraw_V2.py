import numpy as np
import moderngl

class Lines:
    def __init__(self, vat_size, pixels=None):
        if pixels is None:
            pixels = np.zeros((*vat_size, 1)).astype('f4')  # empty screen to start
        self.map_size = vat_size

        self.ctx = moderngl.create_standalone_context(require=430)
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = self.ctx.ADDITIVE_BLENDING

        self.prog = self.ctx.program(
            vertex_shader='''
                        #version 430
                        in vec2 in_vert;
                        in float in_dist;
                        in vec2 in_eng;

                        out float dist;
                        out vec2 eng;

                        uniform vec2 half_resolution;

                        void main() {
                            gl_Position = vec4(vec2(in_vert.xy - half_resolution) / half_resolution, 0.0, 1.0);
                            dist = in_dist;
                            eng = in_eng;
                        }
                    ''',
            fragment_shader='''
                        #version 430
                        in float dist;
                        in vec2 eng;
                        out float f_color;

                        uniform vec2 half_resolution;

                        void main() {
                            vec2 screen_pos = gl_FragCoord.xy;
                            float intensity = eng.x * pow(2.718281828459045f, -dist/eng.y);
                            f_color = intensity;
                        }
                    ''',
        )

        self.prog["half_resolution"] = tuple([_ / 2 for _ in self.map_size])

        self.texture = self.ctx.texture(size=self.map_size, components=1, dtype='f4')
        self.texture.write(pixels.tobytes())
        depth_attachment = self.ctx.depth_renderbuffer(self.map_size)
        self.fbo = self.ctx.framebuffer(self.texture, depth_attachment)
        self.fbo.clear(0.0, 0.0, 0.0)
        self.fbo.use()

    def plotLines(self, arr, pixels=None):
        vbo = self.ctx.buffer(arr.astype('f4'))
        vao = self.ctx.vertex_array(self.prog, [
            (vbo, '2f 1f 2f', 'in_vert', 'in_dist', 'in_eng'),
            # (self.cbo, '4f', 'in_color')
        ])

        if pixels is not None:
            self.texture.write(pixels.tobytes())

        vao.render(moderngl.LINES)
        vao.release()
        vbo.release()
        
    def vat_out(self):
        raw = self.fbo.read(components=1, dtype='f4')
        buf = np.frombuffer(raw, dtype='f4')
        buf = buf.reshape((*self.map_size,))
        return buf


if __name__=="__main__":
    pass

