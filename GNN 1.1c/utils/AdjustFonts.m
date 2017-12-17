function AdjustFonts(handles)
global fonts
for it=fieldnames(handles)'
    if ishandle(handles.(char(it)))
        if ~isempty(strfind(char(it),'Butt')) 
            set(handles.(char(it)),'FontName',fonts.Button.font);
            set(handles.(char(it)),'FontSize',fonts.Button.size);
        elseif  ~isempty(strfind(char(it),'Pop')) 
            set(handles.(char(it)),'FontName',fonts.Pop.font);
            set(handles.(char(it)),'FontSize',fonts.Pop.size);
        elseif  ~isempty(strfind(char(it),'Chk')) 
            set(handles.(char(it)),'FontName',fonts.Chk.font);
            set(handles.(char(it)),'FontSize',fonts.Chk.size);
        elseif ~isempty(strfind(char(it),'Radio')) 
            set(handles.(char(it)),'FontName',fonts.Radio.font);
            set(handles.(char(it)),'FontSize',fonts.Radio.size);
        elseif ~isempty(strfind(char(it),'Edit')) 
            set(handles.(char(it)),'FontName',fonts.Edit.font);
            set(handles.(char(it)),'FontSize',fonts.Edit.size);
        elseif ~isempty(strfind(char(it),'Bar'))
            set(handles.(char(it)),'FontName',fonts.Bar.font);
            set(handles.(char(it)),'FontSize',fonts.Bar.size);
        elseif ~isempty(strfind(char(it),'Text'))
            set(handles.(char(it)),'FontName',fonts.Text.font);
            set(handles.(char(it)),'FontSize',fonts.Text.size);
        elseif ~isempty(strfind(char(it),'Title'))
            set(handles.(char(it)),'FontName',fonts.Title.font);
            set(handles.(char(it)),'FontSize',fonts.Title.size);
        end
    end
end