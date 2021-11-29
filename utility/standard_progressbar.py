from progressbar import ProgressBar, widgets


class StandardProgressBar(ProgressBar):
    def __init__(self, prefix, max_value):
        ProgressBar.__init__(self, prefix=prefix, max_value=max_value, is_terminal=True, term_width=200)
        self.widgets = [widgets.Percentage(**self.widget_kwargs), ' ', widgets.SimpleProgress(format='(%s)' % widgets.SimpleProgress.DEFAULT_FORMAT, **self.widget_kwargs), ' ',
                        widgets.Bar(**self.widget_kwargs), ' ', widgets.Timer(**self.widget_kwargs), ' ', widgets.ETA(**self.widget_kwargs), ]
