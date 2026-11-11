from comfy_api.latest import io


class MixedOutputs(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        template = io.MatchType.Template('value', [io.Image, io.Mask])
        return io.Schema(node_id='MixedOutputs', inputs=[io.MatchType.Input('value', template=template)],
                         outputs=[io.String.Output(), io.MatchType.Output(template=template, tooltip='Matched')])

    @classmethod
    def execute(cls, value):
        return io.NodeOutput('', value)


def test_v3_metadata_preserves_output_positions_and_omits_absent_fields():
    info = MixedOutputs.GET_NODE_INFO_V1()
    assert info['display_name'] == 'MixedOutputs'
    assert info['output_tooltips'] == ['', 'Matched']
    assert info['output_matchtypes'] == ['', 'value']
    assert all(value is not None for value in info.values())
    assert 'price_badge' not in info
    assert 'essentials_category' not in info
