import math


def build_teacher(privileged_dim, hidden_size=128):
    import torch
    from torch import nn
    if privileged_dim < 1 or hidden_size < 16:
        raise ValueError('Invalid privileged teacher dimensions')

    class PrivilegedTeacher(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = dict(schema='privileged-teacher/v1', privileged_dim=privileged_dim,
                               hidden_size=hidden_size, deployable=False)
            self.fusion = nn.Sequential(nn.Linear(privileged_dim + 115, hidden_size),
                                        nn.LayerNorm(hidden_size), nn.SiLU())
            self.belief = nn.GRUCell(hidden_size, hidden_size)
            self.n_head = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.SiLU(), nn.Linear(hidden_size, 5))
            self.m_head = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.SiLU(), nn.Linear(hidden_size, 11))
            self.n_logstd = nn.Parameter(torch.full((3,), math.log(.3)))
            self.m_logstd = nn.Parameter(torch.full((8,), math.log(.2)))
            with torch.no_grad():
                self.n_head[-1].bias[3:] = torch.tensor([2., 0.])
                self.m_head[-1].bias[8:] = torch.tensor([2., 0., 0.])

        def forward(self, privileged, proprio, context, basket, memory, reset=None):
            batch = privileged.shape[0]
            for name, value, width in (('privileged', privileged, privileged_dim), ('proprio', proprio, 85),
                                       ('context', context, 14), ('basket', basket, 16), ('memory', memory, hidden_size)):
                if value.shape != (batch, width):
                    raise ValueError(f'Invalid teacher {name} shape')
            if reset is not None:
                if reset.shape != (batch,) or reset.dtype != torch.bool:
                    raise ValueError('Invalid teacher reset mask')
                memory = torch.where(reset[:, None], torch.zeros_like(memory), memory)
            fused = torch.cat((privileged, proprio, context, basket), dim=-1)
            memory = self.belief(self.fusion(fused), memory)
            n, m = self.n_head(memory), self.m_head(memory)
            return dict(n_mu=n[:, :3], n_events=n[:, 3:], m_mu=m[:, :8], m_events=m[:, 8:],
                        n_logstd=self.n_logstd.clamp(-5., 1.), m_logstd=self.m_logstd.clamp(-5., 1.),
                        memory=memory)

    return PrivilegedTeacher()


def distillation_loss(student, teacher, active, confidence=None):
    import torch
    if active.ndim != 2 or active.shape[1] != 2 or active.dtype != torch.bool:
        raise ValueError('Active masks must be [batch, 2] booleans')
    if confidence is None:
        confidence = torch.ones_like(active, dtype=student['m_mu'].dtype)
    if confidence.shape != active.shape or not torch.isfinite(confidence).all() or (confidence < 0).any() or (confidence > 1).any():
        raise ValueError('Label confidence must be finite in [0, 1]')
    result = {}
    for i, name in enumerate(('n', 'm')):
        weights = active[:, i].to(confidence.dtype) * confidence[:, i].detach()
        action_error = (student[f'{name}_mu'].tanh() - teacher[f'{name}_mu'].detach().tanh()).square().mean(-1)
        target_events = teacher[f'{name}_events'].detach().softmax(-1)
        event_error = -(target_events * student[f'{name}_events'].log_softmax(-1)).sum(-1)
        result[name] = ((action_error + event_error) * weights).sum() / weights.sum().clamp_min(1.)
    result['loss'] = result['n'] + result['m']
    if not torch.isfinite(result['loss']):
        raise RuntimeError('Nonfinite distillation loss')
    return result
