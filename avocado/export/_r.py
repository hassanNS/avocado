from zipfile import ZipFile
from cStringIO import StringIO
from string import punctuation
from django.template import Context
from django.template.loader import get_template
from _base import BaseExporter
from _csv import CSVExporter


class RExporter(BaseExporter):
    short_name = 'R'
    long_name = 'R Programming Language'

    file_extension = 'zip'
    content_type = 'application/zip'

    preferred_formats = ('r', 'coded', 'number', 'string')

    def _format_name(self, name):
        punc = punctuation.replace('_', '')
        name = str(name).translate(None, punc)
        name = name.replace(' ', '_')
        words = name.split('_')
        for i, w in enumerate(words):
            if i == 0:
                name = w.lower()
                continue
            name += w.capitalize()
        if name[0].isdigit():
            name = '_' + name

        return name

    def _code_values(self, name, field):
        "If the field can be coded return the r factor and level for it."
        data_field = 'data${0}'.format(name)

        factor = '{0}.factor = factor({0},levels=c('.format(data_field)
        level = 'levels({0}.factor)=c('.format(data_field)

        values_len = len(field.codes)

        for i, (code, label) in enumerate(field.coded_choices):
            factor += str(code)
            level += '"{0}"'.format(str(label))
            if i == values_len - 1:
                factor += '))\n'
                level += ')\n'
                continue
            factor += ' ,'
            level += ' ,'
        return factor, level

    def write(self, iterable, buff=None, template_name='export/script.R', *args, **kwargs):
        zip_file = ZipFile(self.get_file_obj(buff), 'w')

        factors = []      # field names
        levels = []       # value dictionaries
        labels = []       # data labels

        for c in self.concepts:
            cfields = c.concept_fields.all()
            for cfield in cfields:
                field = cfield.field
                name = self._format_name(field.field_name)
                labels.append('attr(data${0}, "label") = "{1}"'.format(name, str(cfield)))

                if field.lexicon:
                    codes = self._code_values(name, field)
                    factors.append(codes[0])
                    levels.append(codes[1])

        data_filename = 'data.csv'
        script_filename = 'script.R'

        # File buffers
        data_buff = StringIO()
        # Create the data file
        data_exporter = CSVExporter(self.concepts)
        # Overwrite preferred formats for data file
        data_exporter.preferred_formats = self.preferred_formats
        data_exporter.write(iterable, data_buff, *args, **kwargs)

        zip_file.writestr(data_filename, data_buff.getvalue())

        template = get_template(template_name)
        context = Context({
            'data_filename': data_filename,
            'labels': labels,
            'factors': factors,
            'levels': levels,
        })

        # Write script from template
        zip_file.writestr(script_filename, template.render(context))
        zip_file.close()

        return zip_file
